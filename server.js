require('dotenv').config();

const crypto = require('crypto');
const express = require('express');
const cors = require('cors');
const rateLimit = require('express-rate-limit');
const { Pool } = require('pg');
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
const { sendCodeEmail } = require('./mailer');

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
const JWT_SECRET = process.env.JWT_SECRET;
if (!JWT_SECRET || JWT_SECRET.length < 32) {
  console.error('JWT_SECRET is missing or shorter than 32 characters. See .env.example for how to generate one.');
  process.exit(1);
}

const OLLAMA_URL = process.env.OLLAMA_URL || 'http://localhost:11434';
const MODEL_NAME = process.env.MODEL_NAME || 'qwen2.5:7b';
const MAX_HISTORY_MESSAGES = parseInt(process.env.MAX_HISTORY_MESSAGES || '12', 10);
const JWT_EXPIRES_IN = '7d';

const OLLAMA_NUM_CTX = 8192;
const OLLAMA_NUM_PREDICT = 1024;
const OLLAMA_TEMPERATURE = 0.3;
const MAX_MESSAGE_CHARS = 4000;
const MAX_TOTAL_CHARS = 12000;

const CODE_TTL_MIN = 10;
const CODE_MAX_ATTEMPTS = 5;
const RESEND_COOLDOWN_SEC = 60;

// Used so a login for an unknown email takes as long as one for a real email.
const DUMMY_HASH = bcrypt.hashSync('not-a-real-password', 12);

const pool = new Pool({
  host: process.env.PGHOST || 'localhost',
  port: parseInt(process.env.PGPORT || '5432', 10),
  user: process.env.PGUSER || 'postgres',
  password: process.env.PGPASSWORD,
  database: process.env.PGDATABASE || 'chat_app',
  max: 10,
  idleTimeoutMillis: 30000,
});
pool.on('error', (err) => console.error('[pg] idle client error:', err.message));

const app = express();
app.set('trust proxy', 1); // behind ngrok: needed so rate limits see the real client IP
app.use(
  cors({
    // Native apps don't use CORS. Only enable it if you serve a web build.
    origin: process.env.CORS_ORIGIN ? process.env.CORS_ORIGIN.split(',').map((s) => s.trim()) : false,
  })
);
app.use(express.json({ limit: '100kb' }));
app.use((req, res, next) => {
  const start = Date.now();
  res.on('finish', () => {
    console.log(req.method, req.originalUrl.split('?')[0], res.statusCode, `${Date.now() - start}ms`);
  });
  next();
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
const ah = (fn) => (req, res, next) => Promise.resolve(fn(req, res, next)).catch(next);
const httpError = (status, message, extra) => Object.assign(new Error(message), { status, extra });

function parseId(v) {
  const n = Number(v);
  return Number.isInteger(n) && n > 0 && n < 2147483648 ? n : null;
}

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
function normalizeEmail(v) {
  if (typeof v !== 'string') return null;
  const e = v.trim().toLowerCase();
  return e.length <= 254 && EMAIL_RE.test(e) ? e : null;
}

function checkPassword(p) {
  if (typeof p !== 'string' || p.length < 8) throw httpError(400, 'Password must be at least 8 characters');
  if (Buffer.byteLength(p, 'utf8') > 72) throw httpError(400, 'Password is too long (max 72 bytes)');
}

function cleanUsername(v) {
  if (typeof v !== 'string') throw httpError(400, 'Username is required');
  const u = v.trim();
  if (u.length < 2 || u.length > 30) throw httpError(400, 'Username must be 2 to 30 characters');
  return u;
}

function checkCodeFormat(code) {
  if (typeof code !== 'string' || !/^\d{6}$/.test(code)) throw httpError(400, 'Enter the 6-digit code');
}

const USER_COLS = 'id, username, email, display_name, token_version';
const publicUser = (u) => ({ id: u.id, username: u.username, email: u.email, display_name: u.display_name });

function signToken(user) {
  return jwt.sign({ userId: user.id, tv: user.token_version }, JWT_SECRET, {
    algorithm: 'HS256',
    expiresIn: JWT_EXPIRES_IN,
  });
}

function sanitizeHistory(rows) {
  const out = [];
  for (const row of rows) {
    const last = out[out.length - 1];
    if (last && last.role === row.role) {
      last.content += '\n' + row.content;
    } else {
      out.push({ role: row.role, content: row.content });
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Rate limits
// ---------------------------------------------------------------------------
const limiter = (windowMs, limit, message) =>
  rateLimit({ windowMs, limit, standardHeaders: true, legacyHeaders: false, message: { error: message } });

const authLimiter = limiter(15 * 60 * 1000, 20, 'Too many attempts. Try again in a few minutes.');
const codeLimiter = limiter(15 * 60 * 1000, 20, 'Too many code requests. Try again in a few minutes.');
const chatLimiter = rateLimit({
  windowMs: 60 * 1000,
  limit: 20,
  standardHeaders: true,
  legacyHeaders: false,
  keyGenerator: (req) => String(req.user.id),
  message: { error: 'You are sending messages too quickly. Wait a moment.' },
});

// ---------------------------------------------------------------------------
// Auth middleware
// ---------------------------------------------------------------------------
async function requireAuth(req, res, next) {
  const [scheme, token] = (req.header('authorization') || '').split(' ');
  if (scheme !== 'Bearer' || !token) {
    return res.status(401).json({ error: 'Missing or malformed Authorization header' });
  }
  let payload;
  try {
    payload = jwt.verify(token, JWT_SECRET, { algorithms: ['HS256'] });
  } catch {
    return res.status(401).json({ error: 'Invalid or expired token' });
  }
  try {
    const r = await pool.query(
      `SELECT ${USER_COLS}, email_verified FROM users WHERE id = $1`,
      [payload.userId]
    );
    const u = r.rows[0];
    // token_version changes on password/email change, which signs out other devices
    if (!u || !u.email_verified || u.token_version !== payload.tv) {
      return res.status(401).json({ error: 'Session expired. Sign in again.' });
    }
    req.user = publicUser(u);
    next();
  } catch (err) {
    next(err);
  }
}

async function assertOwnsConversation(userId, conversationId) {
  const r = await pool.query('SELECT 1 FROM conversations WHERE id = $1 AND user_id = $2', [
    conversationId,
    userId,
  ]);
  return r.rowCount > 0;
}

// ---------------------------------------------------------------------------
// One-time email codes
// ---------------------------------------------------------------------------
function hashCode(userId, purpose, code) {
  return crypto.createHmac('sha256', JWT_SECRET).update(`${userId}:${purpose}:${code}`).digest('hex');
}

async function issueCode(userId, email, purpose) {
  const cooldown = await pool.query(
    `SELECT CEIL(EXTRACT(EPOCH FROM (created_at + make_interval(secs => $3) - NOW())))::int AS wait
       FROM email_codes
      WHERE user_id = $1 AND purpose = $2
      ORDER BY id DESC LIMIT 1`,
    [userId, purpose, RESEND_COOLDOWN_SEC]
  );
  const wait = cooldown.rows[0]?.wait;
  if (wait > 0) {
    throw httpError(429, `Wait ${wait}s before requesting another code`, { retryAfter: wait });
  }

  const code = String(crypto.randomInt(0, 1000000)).padStart(6, '0');

  // Invalidate older codes for this purpose, then store the new one (hashed).
  await pool.query(
    `UPDATE email_codes SET consumed_at = NOW()
      WHERE user_id = $1 AND purpose = $2 AND consumed_at IS NULL`,
    [userId, purpose]
  );
  await pool.query(
    `INSERT INTO email_codes (user_id, email, purpose, code_hash, expires_at)
     VALUES ($1, $2, $3, $4, NOW() + make_interval(mins => $5))`,
    [userId, email, purpose, hashCode(userId, purpose, code), CODE_TTL_MIN]
  );

  try {
    await sendCodeEmail(email, code, purpose, CODE_TTL_MIN);
  } catch (err) {
    console.error('[mail] send failed:', err.message);
    throw httpError(502, 'Could not send the email. Try again shortly.');
  }
}

// Returns the code row ({ email }) on success; throws on any failure.
async function consumeCode(userId, purpose, code) {
  const r = await pool.query(
    `SELECT id, email, code_hash FROM email_codes
      WHERE user_id = $1 AND purpose = $2 AND consumed_at IS NULL AND expires_at > NOW()
      ORDER BY id DESC LIMIT 1`,
    [userId, purpose]
  );
  const row = r.rows[0];
  if (!row) throw httpError(400, 'Code expired or not requested. Request a new one.');

  // Count the attempt first (atomic) so parallel guesses can't exceed the limit.
  const bump = await pool.query(
    `UPDATE email_codes SET attempts = attempts + 1
      WHERE id = $1 AND attempts < $2 RETURNING attempts`,
    [row.id, CODE_MAX_ATTEMPTS]
  );
  if (bump.rowCount === 0) throw httpError(429, 'Too many wrong attempts. Request a new code.');

  const given = Buffer.from(hashCode(userId, purpose, code), 'hex');
  const stored = Buffer.from(row.code_hash, 'hex');
  if (given.length !== stored.length || !crypto.timingSafeEqual(given, stored)) {
    const left = CODE_MAX_ATTEMPTS - bump.rows[0].attempts;
    throw httpError(400, left > 0 ? `Incorrect code. ${left} attempt${left === 1 ? '' : 's'} left.` : 'Incorrect code. Request a new one.');
  }

  const used = await pool.query(
    'UPDATE email_codes SET consumed_at = NOW() WHERE id = $1 AND consumed_at IS NULL RETURNING id',
    [row.id]
  );
  if (used.rowCount === 0) throw httpError(400, 'Code already used. Request a new one.');
  return { email: row.email };
}

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------
app.get('/health', async (req, res) => {
  try {
    await pool.query('SELECT 1');
    res.json({ status: 'ok', db: 'connected' });
  } catch (err) {
    console.error('[health]', err.message);
    res.status(500).json({ status: 'error' });
  }
});

// ---------------------------------------------------------------------------
// Auth
// ---------------------------------------------------------------------------
app.post(
  '/api/auth/signup',
  authLimiter,
  ah(async (req, res) => {
    const username = cleanUsername(req.body.username);
    const email = normalizeEmail(req.body.email);
    if (!email) throw httpError(400, 'Enter a valid email address');
    checkPassword(req.body.password);

    const hash = await bcrypt.hash(req.body.password, 12);

    let userId;
    const existing = await pool.query('SELECT id, email_verified FROM users WHERE lower(email) = $1', [email]);
    if (existing.rowCount > 0) {
      if (existing.rows[0].email_verified) {
        throw httpError(409, 'An account with this email already exists');
      }
      // Earlier signup never verified: let the new attempt replace it.
      userId = existing.rows[0].id;
      await pool.query(
        'UPDATE users SET username = $1, display_name = $1, password_hash = $2 WHERE id = $3',
        [username, hash, userId]
      );
    } else {
      try {
        const ins = await pool.query(
          `INSERT INTO users (username, email, password_hash, display_name, email_verified)
           VALUES ($1, $2, $3, $1, FALSE) RETURNING id`,
          [username, email, hash]
        );
        userId = ins.rows[0].id;
      } catch (err) {
        if (err.code === '23505') throw httpError(409, 'An account with this email already exists');
        throw err;
      }
    }

    await issueCode(userId, email, 'verify_email');
    res.status(201).json({ needsVerification: true, email });
  })
);

app.post(
  '/api/auth/verify-email',
  codeLimiter,
  ah(async (req, res) => {
    const email = normalizeEmail(req.body.email);
    checkCodeFormat(req.body.code);
    if (!email) throw httpError(400, 'Invalid or expired code');

    const r = await pool.query('SELECT id FROM users WHERE lower(email) = $1 AND email_verified = FALSE', [email]);
    if (r.rowCount === 0) throw httpError(400, 'Invalid or expired code');

    await consumeCode(r.rows[0].id, 'verify_email', req.body.code);
    const u = await pool.query(
      `UPDATE users SET email_verified = TRUE WHERE id = $1 RETURNING ${USER_COLS}`,
      [r.rows[0].id]
    );
    res.json({ token: signToken(u.rows[0]), user: publicUser(u.rows[0]) });
  })
);

app.post(
  '/api/auth/resend',
  codeLimiter,
  ah(async (req, res) => {
    const email = normalizeEmail(req.body.email);
    if (email) {
      const r = await pool.query('SELECT id FROM users WHERE lower(email) = $1 AND email_verified = FALSE', [email]);
      if (r.rowCount > 0) {
        try {
          await issueCode(r.rows[0].id, email, 'verify_email');
        } catch (err) {
          if (err.status !== 429) throw err;
        }
      }
    }
    // Same answer whether or not the email exists, so this can't be used to probe for accounts.
    res.json({ ok: true });
  })
);

app.post(
  '/api/auth/login',
  authLimiter,
  ah(async (req, res) => {
    const email = normalizeEmail(req.body.email);
    const password = typeof req.body.password === 'string' ? req.body.password : '';
    if (!email || !password) throw httpError(400, 'Email and password are required');

    const r = await pool.query(
      `SELECT ${USER_COLS}, password_hash, email_verified FROM users WHERE lower(email) = $1`,
      [email]
    );
    const user = r.rows[0];
    const ok = password.length <= 200 && (await bcrypt.compare(password, user ? user.password_hash : DUMMY_HASH));
    if (!user || !ok) throw httpError(401, 'Invalid email or password');

    if (!user.email_verified) {
      try {
        await issueCode(user.id, user.email, 'verify_email');
      } catch (err) {
        if (err.status !== 429) throw err;
      }
      throw httpError(403, 'Verify your email to continue', { code: 'EMAIL_NOT_VERIFIED', email: user.email });
    }
    res.json({ token: signToken(user), user: publicUser(user) });
  })
);

app.get('/api/me', requireAuth, (req, res) => res.json({ user: req.user }));

// ---------------------------------------------------------------------------
// Account changes (password / email need a code; username does not)
// ---------------------------------------------------------------------------
app.post(
  '/api/account/request-code',
  requireAuth,
  codeLimiter,
  ah(async (req, res) => {
    const { purpose } = req.body;

    if (purpose === 'change_password') {
      await issueCode(req.user.id, req.user.email, 'change_password');
      return res.json({ ok: true, sentTo: req.user.email });
    }

    if (purpose === 'change_email') {
      const newEmail = normalizeEmail(req.body.newEmail);
      if (!newEmail) throw httpError(400, 'Enter a valid email address');
      if (newEmail === req.user.email) throw httpError(400, 'That is already your email');

      const current = await pool.query('SELECT password_hash FROM users WHERE id = $1', [req.user.id]);
      const pw = typeof req.body.currentPassword === 'string' ? req.body.currentPassword : '';
      if (!pw || pw.length > 200 || !(await bcrypt.compare(pw, current.rows[0].password_hash))) {
        throw httpError(403, 'Current password is incorrect');
      }

      const taken = await pool.query('SELECT 1 FROM users WHERE lower(email) = $1', [newEmail]);
      if (taken.rowCount > 0) throw httpError(409, 'That email is already in use');

      await issueCode(req.user.id, newEmail, 'change_email'); // sent to the NEW address
      return res.json({ ok: true, sentTo: newEmail });
    }

    throw httpError(400, 'Invalid purpose');
  })
);

app.post(
  '/api/account/change-password',
  requireAuth,
  codeLimiter,
  ah(async (req, res) => {
    checkCodeFormat(req.body.code);
    checkPassword(req.body.newPassword); // validate before spending the code

    await consumeCode(req.user.id, 'change_password', req.body.code);
    const hash = await bcrypt.hash(req.body.newPassword, 12);
    const u = await pool.query(
      `UPDATE users SET password_hash = $1, token_version = token_version + 1
        WHERE id = $2 RETURNING ${USER_COLS}`,
      [hash, req.user.id]
    );
    // New token keeps this device signed in; every other device is signed out.
    res.json({ token: signToken(u.rows[0]), user: publicUser(u.rows[0]) });
  })
);

app.post(
  '/api/account/change-email',
  requireAuth,
  codeLimiter,
  ah(async (req, res) => {
    checkCodeFormat(req.body.code);
    const { email: newEmail } = await consumeCode(req.user.id, 'change_email', req.body.code);
    try {
      const u = await pool.query(
        `UPDATE users SET email = $1, email_verified = TRUE, token_version = token_version + 1
          WHERE id = $2 RETURNING ${USER_COLS}`,
        [newEmail, req.user.id]
      );
      res.json({ token: signToken(u.rows[0]), user: publicUser(u.rows[0]) });
    } catch (err) {
      if (err.code === '23505') throw httpError(409, 'That email is already in use');
      throw err;
    }
  })
);

app.patch(
  '/api/account/username',
  requireAuth,
  ah(async (req, res) => {
    const username = cleanUsername(req.body.username);
    const u = await pool.query(
      `UPDATE users SET username = $1, display_name = $1 WHERE id = $2 RETURNING ${USER_COLS}`,
      [username, req.user.id]
    );
    res.json({ user: publicUser(u.rows[0]) });
  })
);

// ---------------------------------------------------------------------------
// Conversations
// ---------------------------------------------------------------------------
app.get(
  '/api/conversations',
  requireAuth,
  ah(async (req, res) => {
    const result = await pool.query(
      `SELECT id, title, created_at, updated_at
         FROM conversations WHERE user_id = $1 ORDER BY updated_at DESC`,
      [req.user.id]
    );
    res.json({ conversations: result.rows });
  })
);

app.post(
  '/api/conversations',
  requireAuth,
  ah(async (req, res) => {
    const title = (typeof req.body.title === 'string' ? req.body.title.trim() : '').slice(0, 100) || 'New chat';
    const result = await pool.query(
      `INSERT INTO conversations (user_id, title) VALUES ($1, $2)
       RETURNING id, title, created_at, updated_at`,
      [req.user.id, title]
    );
    res.status(201).json({ conversation: result.rows[0] });
  })
);

app.get(
  '/api/conversations/:id/messages',
  requireAuth,
  ah(async (req, res) => {
    const id = parseId(req.params.id);
    if (!id || !(await assertOwnsConversation(req.user.id, id))) throw httpError(404, 'Not found');
    const result = await pool.query(
      `SELECT id, role, content, created_at FROM messages WHERE conversation_id = $1 ORDER BY id ASC`,
      [id]
    );
    res.json({ messages: result.rows });
  })
);

app.put(
  '/api/conversations/:id',
  requireAuth,
  ah(async (req, res) => {
    const id = parseId(req.params.id);
    const title = (typeof req.body.title === 'string' ? req.body.title.trim() : '').slice(0, 100);
    if (!title) throw httpError(400, 'title is required');
    if (!id || !(await assertOwnsConversation(req.user.id, id))) throw httpError(404, 'Not found');
    const result = await pool.query(
      `UPDATE conversations SET title = $1, updated_at = NOW() WHERE id = $2 RETURNING id, title, updated_at`,
      [title, id]
    );
    res.json({ conversation: result.rows[0] });
  })
);

app.delete(
  '/api/conversations/:id',
  requireAuth,
  ah(async (req, res) => {
    const id = parseId(req.params.id);
    if (!id || !(await assertOwnsConversation(req.user.id, id))) throw httpError(404, 'Not found');
    await pool.query('DELETE FROM conversations WHERE id = $1', [id]);
    res.json({ deleted: id });
  })
);

// ---------------------------------------------------------------------------
// Chat (streaming SSE)
// ---------------------------------------------------------------------------

// Yields one parsed JSON object per line, buffering partial lines across network chunks.
async function* ollamaStream(messages, signal) {
  const r = await fetch(`${OLLAMA_URL}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    signal,
    body: JSON.stringify({
      model: MODEL_NAME,
      messages,
      stream: true,
      options: {
        num_ctx: OLLAMA_NUM_CTX,
        num_predict: OLLAMA_NUM_PREDICT,
        temperature: OLLAMA_TEMPERATURE,
        repeat_penalty: 1.1,
      },
    }),
  });
  if (!r.ok) {
    const body = await r.text().catch(() => '');
    throw new Error(`Ollama responded ${r.status}: ${body.slice(0, 200)}`);
  }

  const decoder = new TextDecoder(); // stream mode keeps multi-byte characters intact across chunks
  let buf = '';
  const parse = (line) => {
    try {
      return JSON.parse(line);
    } catch {
      return null;
    }
  };

  for await (const chunk of r.body) {
    buf += decoder.decode(chunk, { stream: true });
    let nl;
    while ((nl = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (!line) continue;
      const parsed = parse(line);
      if (!parsed) continue;
      if (parsed.error) throw new Error(parsed.error);
      yield parsed;
    }
  }
  const tail = (buf + decoder.decode()).trim();
  if (tail) {
    const parsed = parse(tail);
    if (parsed) {
      if (parsed.error) throw new Error(parsed.error);
      yield parsed;
    }
  }
}

function tailRepeats(text) {
  const tail = text.slice(-200);
  return text.slice(0, -200).includes(tail);
}

function looksLikeLoop(text) {
  const words = text.toLowerCase().split(/\s+/);
  return words.length > 50 && new Set(words).size / words.length < 0.15;
}

app.post(
  '/api/chat',
  requireAuth,
  chatLimiter,
  ah(async (req, res) => {
    const conversationId = parseId(req.body.conversationId);
    const message = typeof req.body.message === 'string' ? req.body.message.trim() : '';
    if (!conversationId || !message) throw httpError(400, 'conversationId and message are required');
    if (message.length > MAX_MESSAGE_CHARS) {
      throw httpError(400, `Message is too long (max ${MAX_MESSAGE_CHARS} characters)`);
    }
    if (!(await assertOwnsConversation(req.user.id, conversationId))) throw httpError(404, 'Not found');

    await pool.query(`INSERT INTO messages (conversation_id, role, content) VALUES ($1, 'user', $2)`, [
      conversationId,
      message,
    ]);
    await pool.query('UPDATE conversations SET updated_at = NOW() WHERE id = $1', [conversationId]);

    const historyResult = await pool.query(
      `SELECT role, content FROM (
         SELECT id, role, content FROM messages
          WHERE conversation_id = $1 ORDER BY id DESC LIMIT $2
       ) sub ORDER BY id ASC`,
      [conversationId, MAX_HISTORY_MESSAGES]
    );

    // Trim each message, then keep as many recent ones as fit in MAX_TOTAL_CHARS.
    let totalChars = 0;
    const cappedHistory = [];
    for (let i = historyResult.rows.length - 1; i >= 0; i--) {
      const row = historyResult.rows[i];
      const content =
        row.content.length > MAX_MESSAGE_CHARS
          ? row.content.slice(0, MAX_MESSAGE_CHARS) + '\n\n[...truncated...]'
          : row.content;
      if (totalChars + content.length > MAX_TOTAL_CHARS) break;
      cappedHistory.unshift({ role: row.role, content });
      totalChars += content.length;
    }

    const displayName = (req.user.display_name || req.user.username || 'the user')
      .replace(/["\r\n]/g, ' ')
      .slice(0, 50);

    const systemMessage = {
      role: 'system',
      content:
        'You are a helpful personal assistant in a mobile chat app. ' +
        `The user's name is "${displayName}". ` +
        'Always reply in English unless the user explicitly asks for another language. ' +
        'You DO have access to the conversation history that appears before this message. ' +
        'When the user asks what was said earlier, look through the messages above and answer directly. ' +
        'Do not say you lack access to past messages — you can see them. ' +
        'Do not tell the user to check their own chat history. ' +
        'Keep answers on-topic. ' +
        'Do not loop, repeat yourself, or drift into unrelated topics.',
    };
    const history = [systemMessage, ...sanitizeHistory(cappedHistory)];

    res.set({
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
      'X-Accel-Buffering': 'no',
    });
    res.flushHeaders();

    let clientGone = false;
    let current = null; // AbortController for the Ollama request in flight
    res.on('close', () => {
      if (!res.writableFinished) {
        clientGone = true;
        current?.abort(); // stop generating if the app was closed or the request cancelled
      }
    });

    const send = (obj) => {
      if (!clientGone && !res.writableEnded) res.write(`data: ${JSON.stringify(obj)}\n\n`);
    };

    async function generate(msgs) {
      current = new AbortController();
      let reply = '';
      let lastCheck = 0;
      let repeated = false;
      for await (const p of ollamaStream(msgs, current.signal)) {
        const piece = p.message?.content || '';
        if (!piece) continue;
        reply += piece;
        send({ content: piece });
        if (reply.length > 800 && reply.length - lastCheck >= 100) {
          lastCheck = reply.length;
          if (tailRepeats(reply)) {
            repeated = true;
            break;
          }
        }
      }
      try {
        current.abort(); // release the connection
      } catch {}
      return { reply, repeated };
    }

    try {
      let result = await generate(history);

      if (!clientGone && !result.repeated && result.reply.trim() === '') {
        // Retry once with a minimal prompt so a poisoned history can't block simple messages.
        console.error('[ollama] empty reply, retrying with minimal history');
        result = await generate([systemMessage, { role: 'user', content: message }]);
      }

      if (!clientGone) {
        if (result.repeated || looksLikeLoop(result.reply)) {
          console.error('[ollama] repetition detected, reply not saved');
          send({ error: 'Generation stopped (repetition detected)' });
        } else if (result.reply.trim() === '') {
          send({ error: 'Model returned no response' });
        } else {
          await pool.query(`INSERT INTO messages (conversation_id, role, content) VALUES ($1, 'assistant', $2)`, [
            conversationId,
            result.reply,
          ]);
          send({ done: true });
        }
      }
    } catch (err) {
      if (!clientGone) {
        console.error('[chat]', err.message);
        send({ error: 'The model failed to respond. Try again.' });
      }
    } finally {
      if (!res.writableEnded) res.end();
    }
  })
);

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------
app.use((err, req, res, next) => {
  if (res.headersSent) return next(err);
  if (err.type === 'entity.parse.failed') return res.status(400).json({ error: 'Invalid JSON' });
  if (err.type === 'entity.too.large') return res.status(413).json({ error: 'Request too large' });
  const status = err.status || 500;
  if (status >= 500) console.error(err);
  res.status(status).json({ error: status >= 500 && status !== 502 ? 'Server error' : err.message, ...(err.extra || {}) });
});

process.on('unhandledRejection', (err) => console.error('[unhandledRejection]', err));

// Housekeeping: drop signups that never verified and old codes.
async function cleanup() {
  try {
    await pool.query(
      `DELETE FROM users WHERE email_verified = FALSE AND created_at < NOW() - INTERVAL '1 day'`
    );
    await pool.query(`DELETE FROM email_codes WHERE expires_at < NOW() - INTERVAL '1 day'`);
  } catch (err) {
    console.error('[cleanup]', err.message);
  }
}
setInterval(cleanup, 60 * 60 * 1000).unref();
cleanup();

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Chat backend running on http://localhost:${PORT}`);
  console.log(`Forwarding to Ollama at ${OLLAMA_URL} using model ${MODEL_NAME}`);
  console.log(`  num_ctx=${OLLAMA_NUM_CTX} num_predict=${OLLAMA_NUM_PREDICT}`);
  console.log(`PostgreSQL ${process.env.PGHOST || 'localhost'}:${process.env.PGPORT || 5432}/${process.env.PGDATABASE || 'chat_app'}`);
  if (!process.env.SMTP_USER) console.log('SMTP not configured: verification codes will be printed here.');
});