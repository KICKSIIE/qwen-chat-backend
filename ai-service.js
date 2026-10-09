/**
 * AI service: the only program that talks to Ollama.
 *
 * It offers an OpenAI-style API (/v1) protected by one shared key (AI_API_KEY).
 * It knows nothing about users, conversations or the database. Callers (the CRUD server,
 * or any other AI client) send the messages they want answered, and get the reply back.
 *
 * Run it on its own:  node ai-service.js
 * Default address:    http://localhost:3001  (change with AI_PORT)
 *
 * To let other PCs use it, expose it with a tunnel (for example: ngrok http 3001).
 */
require('dotenv').config();

const crypto = require('crypto');
const express = require('express');
const rateLimit = require('express-rate-limit');

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
const PORT = process.env.AI_PORT || 3001;

// The one shared key callers must send ("Authorization: Bearer <key>").
// Anyone who has it can use the model, so give it only to people you trust.
// Generate one with: node -e "console.log('sk-' + require('crypto').randomBytes(32).toString('hex'))"
const AI_API_KEY = process.env.AI_API_KEY || '';
const AI_API_KEY_ENABLED = AI_API_KEY.length >= 24; // refuse short, guessable keys

// ---- Branding ----
// The name the AI uses for itself, and the model name shown to API clients.
// Callers never see the real model name (MODEL_NAME below), only PUBLIC_MODEL_NAME.
const AI_NAME = (process.env.AI_NAME || 'MashfyAI').replace(/["\r\n]/g, ' ').trim().slice(0, 40) || 'MashfyAI';
const PUBLIC_MODEL_NAME =
  process.env.PUBLIC_MODEL_NAME || AI_NAME.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '') || 'ai';
// Added to the start of every conversation, whoever the caller is.
const BRAND_PROMPT =
  `Your name is ${AI_NAME}. When you introduce yourself or are asked who you are, say that you are ${AI_NAME}. ` +
  `If asked who made you or what powers you, say you are ${AI_NAME}, an AI assistant built on an open-source language model. ` +
  `Do not call yourself Qwen, ChatGPT, Claude, Gemini or any other assistant name.`;

const OLLAMA_URL = process.env.OLLAMA_URL || 'http://localhost:11434';
// NOTE: photos only work with a vision-capable model. A text-only model such as
// qwen2.5:7b silently ignores images, so answers won't relate to the photo.
const MODEL_NAME = process.env.MODEL_NAME || 'qwen2.5vl:7b';
const OLLAMA_NUM_CTX = 8192;
const OLLAMA_NUM_PREDICT = 1024; // hard cap on reply length (in tokens)
const OLLAMA_TEMPERATURE = 0.3;
// Overall time limit for one generation, so a hung Ollama can't keep a request open forever.
const OLLAMA_TIMEOUT_MS = 4 * 60 * 1000;

const MAX_IMAGE_B64_CHARS = 3000000; // about 2.2 MB of JPEG
const MAX_MESSAGES = 50;
const MAX_TOTAL_CHARS = 24000; // total text characters in one request
// How many replies may be generated at once. Extra requests get a "busy" 429.
const MAX_CONCURRENT = parseInt(process.env.AI_MAX_CONCURRENT || '6', 10);
// Requests per minute per IP. If the CRUD server calls from the same PC, all its users share
// one IP, so this must be comfortably higher than the sum of their own limits.
const RATE_LIMIT_PER_MIN = parseInt(process.env.AI_RATE_LIMIT_PER_MIN || '120', 10);

let active = 0; // replies being generated right now

const app = express();
// Number of proxies in front of this service (1 when it is behind ngrok, 0 when called directly).
// A wrong value lets clients fake their IP and dodge the rate limit.
app.set('trust proxy', parseInt(process.env.TRUST_PROXY ?? '1', 10));

// Simple request log: method, path, status, duration.
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

// Errors in the shape OpenAI clients expect: { error: { message, type, code } }.
const oaiError = (res, status, message, type = 'invalid_request_error', code = null) =>
  res.status(status).json({ error: { message, type, code } });

// Compares keys in constant time (hashing first makes both the same length).
const sha = (s) => crypto.createHash('sha256').update(s).digest();
const keysMatch = (given, expected) => crypto.timingSafeEqual(sha(given), sha(expected));

// Per-IP limit. It runs BEFORE the key check so wrong-key guessing is throttled too.
const limiter = rateLimit({
  windowMs: 60 * 1000,
  limit: RATE_LIMIT_PER_MIN,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: { message: 'Too many requests. Wait a moment.', type: 'rate_limit_error', code: null } },
});

function requireKey(req, res, next) {
  if (!AI_API_KEY_ENABLED) {
    return oaiError(res, 503, 'This AI service has no API key configured.', 'server_error');
  }
  const [scheme, token] = (req.header('authorization') || '').split(' ');
  if (scheme !== 'Bearer' || !token || token.length > 300 || !keysMatch(token, AI_API_KEY)) {
    return oaiError(res, 401, 'Incorrect API key provided.', 'invalid_request_error', 'invalid_api_key');
  }
  next();
}

// ---------------------------------------------------------------------------
// Talking to Ollama
// ---------------------------------------------------------------------------

// Yields one parsed JSON object per line, buffering partial lines across network chunks.
async function* ollamaStream(messages, signal, extraOptions = {}) {
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
        ...extraOptions, // lets a request lower num_predict or change temperature
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
  // Whatever is left after the stream ends (a final line with no trailing newline).
  const tail = (buf + decoder.decode()).trim();
  if (tail) {
    const parsed = parse(tail);
    if (parsed) {
      if (parsed.error) throw new Error(parsed.error);
      yield parsed;
    }
  }
}

// Loop detectors for small models that sometimes repeat themselves forever.
// 1) the last 200 characters already appeared earlier in the reply.
function tailRepeats(text) {
  const tail = text.slice(-200);
  return text.slice(0, -200).includes(tail);
}

// 2) very few distinct words compared with the total word count.
function looksLikeLoop(text) {
  const words = text.toLowerCase().split(/\s+/);
  return words.length > 50 && new Set(words).size / words.length < 0.15;
}

// Puts the brand identity at the very start. It is merged into the first system message
// (if there is one) instead of added as a second one, because some models ignore any system
// message that is not first.
function withBranding(msgs) {
  const out = msgs.map((m) => ({ ...m }));
  if (out[0] && out[0].role === 'system') {
    out[0].content = out[0].content ? `${BRAND_PROMPT}\n\n${out[0].content}` : BRAND_PROMPT;
  } else {
    out.unshift({ role: 'system', content: BRAND_PROMPT });
  }
  return out;
}

// Runs one generation and reports each piece of text through onPiece().
// Returns the text plus flags: `repeated` (stopped for looping), `truncated`
// (hit the OLLAMA_NUM_PREDICT limit) and token counts.
async function generateReply(msgs, signal, onPiece, extraOptions = {}) {
  const ctl = new AbortController();
  const onAbort = () => ctl.abort(); // the caller left: stop generating
  if (signal.aborted) ctl.abort();
  signal.addEventListener('abort', onAbort);
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    ctl.abort();
  }, OLLAMA_TIMEOUT_MS);

  let reply = '';
  let lastCheck = 0;
  let repeated = false;
  let truncated = false;
  let usage = { prompt: 0, completion: 0 };
  try {
    for await (const p of ollamaStream(msgs, ctl.signal, extraOptions)) {
      // The final chunk has no text but says why generation ended.
      if (p.done) {
        if (p.done_reason === 'length') truncated = true;
        usage = { prompt: p.prompt_eval_count || 0, completion: p.eval_count || 0 };
      }
      const piece = p.message?.content || '';
      if (!piece) continue;
      reply += piece;
      onPiece(piece);
      // Check for loops every ~100 new characters, not on every token.
      if (reply.length > 800 && reply.length - lastCheck >= 100) {
        lastCheck = reply.length;
        if (tailRepeats(reply)) {
          repeated = true;
          break;
        }
      }
    }
  } catch (err) {
    // An abort because the caller left is not an error; an abort from our own timer is.
    if (err.name !== 'AbortError') throw err;
    if (timedOut) throw new Error('Ollama timed out');
  } finally {
    clearTimeout(timer);
    signal.removeEventListener('abort', onAbort);
    ctl.abort(); // makes sure the connection to Ollama is closed
  }
  return { reply, repeated, truncated, usage };
}

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

// No key needed: lets you check that the service and Ollama are running.
app.get('/health', async (req, res) => {
  let ollama = 'down';
  try {
    const r = await fetch(`${OLLAMA_URL}/api/tags`, { signal: AbortSignal.timeout(2000) });
    if (r.ok) ollama = 'up';
  } catch {}
  res.json({ status: 'ok', ollama, model: PUBLIC_MODEL_NAME });
});

// Lists the model, so clients can fill their model picker.
app.get('/v1/models', limiter, requireKey, (req, res) => {
  res.json({ object: 'list', data: [{ id: PUBLIC_MODEL_NAME, object: 'model', created: 0, owned_by: 'local' }] });
});

// Order: rate limit, key check, THEN the big body parser, so people without the key
// can't make us buffer large bodies.
app.post(
  '/v1/chat/completions',
  limiter,
  requireKey,
  express.json({ limit: '12mb' }), // room for one downsized photo
  ah(async (req, res) => {
    const body = req.body || {};
    const msgs = body.messages;
    if (!Array.isArray(msgs) || msgs.length === 0 || msgs.length > MAX_MESSAGES) {
      return oaiError(res, 400, `messages must be an array of 1 to ${MAX_MESSAGES} items.`);
    }

    // Convert OpenAI messages into Ollama messages.
    const ollamaMsgs = [];
    let totalChars = 0;
    let imageCount = 0;
    for (const m of msgs) {
      if (!m || !['system', 'user', 'assistant'].includes(m.role)) {
        return oaiError(res, 400, 'Each message needs a role of system, user or assistant.');
      }
      let text = '';
      const images = [];
      if (typeof m.content === 'string') {
        text = m.content;
      } else if (Array.isArray(m.content)) {
        for (const part of m.content) {
          if (part && part.type === 'text' && typeof part.text === 'string') {
            text += (text ? '\n' : '') + part.text;
          } else if (part && part.type === 'image_url') {
            const url = typeof part.image_url === 'string' ? part.image_url : part.image_url?.url;
            const match = typeof url === 'string' ? url.match(/^data:image\/jpe?g;base64,([A-Za-z0-9+/]+={0,2})$/) : null;
            // Only inline JPEGs: this service never downloads images from web links.
            if (!match || match[1].length > MAX_IMAGE_B64_CHARS || !match[1].startsWith('/9j/')) {
              return oaiError(res, 400, 'Images must be JPEG data URLs (data:image/jpeg;base64,...) up to about 2 MB. Web links are not fetched.');
            }
            images.push(match[1]);
            imageCount++;
          } else {
            return oaiError(res, 400, 'Unsupported content part. Use type "text" or "image_url".');
          }
        }
      } else if (m.content != null) {
        return oaiError(res, 400, 'content must be a string or an array of parts.');
      }
      totalChars += text.length;
      ollamaMsgs.push({ role: m.role, content: text, ...(images.length ? { images } : {}) });
    }
    if (imageCount > 1) return oaiError(res, 400, 'Only one image per request is supported.');
    if (totalChars > MAX_TOTAL_CHARS) {
      return oaiError(res, 400, `Messages are too long (max ${MAX_TOTAL_CHARS} characters in total).`);
    }
    if (!ollamaMsgs.some((m) => m.role === 'user' && (m.content.trim() || m.images))) {
      return oaiError(res, 400, 'Include at least one non-empty user message.');
    }

    // Callers may ask for settings, but they are clamped to this service's limits.
    // The model name they send is ignored: this service always answers with its own model.
    const opts = {};
    const maxTokens = Number(body.max_tokens ?? body.max_completion_tokens);
    if (Number.isFinite(maxTokens) && maxTokens > 0) opts.num_predict = Math.min(Math.floor(maxTokens), OLLAMA_NUM_PREDICT);
    const temp = Number(body.temperature);
    if (body.temperature != null && Number.isFinite(temp)) opts.temperature = Math.min(Math.max(temp, 0), 1.5);

    if (active >= MAX_CONCURRENT) {
      res.set('Retry-After', '5');
      return oaiError(res, 429, 'The AI is busy. Try again in a few seconds.', 'rate_limit_error');
    }

    const id = 'chatcmpl-' + crypto.randomBytes(12).toString('hex');
    const created = Math.floor(Date.now() / 1000);
    const ac = new AbortController();
    let clientGone = false;
    res.on('close', () => {
      if (!res.writableFinished) {
        clientGone = true;
        ac.abort(); // the caller disconnected: stop generating
      }
    });

    const branded = withBranding(ollamaMsgs);

    active++;
    try {
      if (body.stream === true) {
        // Streaming: "data: {chunk}" lines, ending with "data: [DONE]".
        res.set({
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          Connection: 'keep-alive',
          'X-Accel-Buffering': 'no',
        });
        res.flushHeaders();
        const chunk = (delta, finish = null) =>
          res.write(
            `data: ${JSON.stringify({
              id,
              object: 'chat.completion.chunk',
              created,
              model: PUBLIC_MODEL_NAME,
              choices: [{ index: 0, delta, finish_reason: finish }],
            })}\n\n`
          );
        const streamError = (message) =>
          res.write(`data: ${JSON.stringify({ error: { message, type: 'server_error', code: null } })}\n\n`);
        try {
          chunk({ role: 'assistant' });
          const r = await generateReply(branded, ac.signal, (piece) => {
            if (!clientGone) chunk({ content: piece });
          }, opts);
          if (!clientGone) {
            if (r.repeated || looksLikeLoop(r.reply)) {
              // Tell the caller instead of finishing normally, so it can discard the looping text.
              console.error('[ai] repetition detected');
              streamError('Generation stopped (repetition detected).');
            } else {
              chunk({}, r.truncated ? 'length' : 'stop');
              res.write('data: [DONE]\n\n');
            }
          }
        } catch (err) {
          console.error('[ai]', err.message);
          if (!clientGone) streamError('The model failed to respond. Try again.');
        } finally {
          if (!res.writableEnded) res.end();
        }
        return;
      }

      // Not streaming: wait for the whole reply, then send one JSON object.
      let r;
      try {
        r = await generateReply(branded, ac.signal, () => {}, opts);
      } catch (err) {
        console.error('[ai]', err.message);
        if (!clientGone) oaiError(res, 502, 'The model failed to respond. Try again.', 'server_error');
        return;
      }
      if (clientGone) return;
      if (r.repeated || looksLikeLoop(r.reply)) {
        return oaiError(res, 502, 'Generation stopped (repetition detected).', 'server_error');
      }
      if (r.reply.trim() === '') return oaiError(res, 502, 'Model returned no response.', 'server_error');
      res.json({
        id,
        object: 'chat.completion',
        created,
        model: PUBLIC_MODEL_NAME,
        choices: [
          {
            index: 0,
            message: { role: 'assistant', content: r.reply },
            finish_reason: r.truncated ? 'length' : 'stop',
          },
        ],
        usage: {
          prompt_tokens: r.usage.prompt,
          completion_tokens: r.usage.completion,
          total_tokens: r.usage.prompt + r.usage.completion,
        },
      });
    } finally {
      active--;
    }
  })
);

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------
app.use((req, res) => oaiError(res, 404, 'Not found.'));

app.use((err, req, res, next) => {
  if (res.headersSent) return next(err);
  const bad = err.type === 'entity.parse.failed';
  const big = err.type === 'entity.too.large';
  const code = bad ? 400 : big ? 413 : err.status || 500;
  if (code >= 500) console.error(err);
  const message = bad ? 'Invalid JSON' : big ? 'Request too large' : code >= 500 ? 'Server error' : err.message;
  oaiError(res, code, message, code >= 500 ? 'server_error' : 'invalid_request_error');
});

process.on('unhandledRejection', (err) => console.error('[unhandledRejection]', err));

app.listen(PORT, () => {
  console.log(`${AI_NAME} service running on http://localhost:${PORT} (public model name: ${PUBLIC_MODEL_NAME})`);
  console.log(`Forwarding to Ollama at ${OLLAMA_URL} using model ${MODEL_NAME}`);
  console.log(`  num_ctx=${OLLAMA_NUM_CTX} num_predict=${OLLAMA_NUM_PREDICT}`);
  console.log(
    AI_API_KEY_ENABLED
      ? 'API key: set (callers must send it)'
      : 'WARNING: AI_API_KEY is missing or shorter than 24 characters. Every /v1 request will be refused.'
  );
});