const nodemailer = require('nodemailer');

const SMTP_USER = process.env.SMTP_USER;
const SMTP_PASS = (process.env.SMTP_PASS || '').replace(/\s+/g, ''); // Google shows app passwords with spaces
const SMTP_PORT = parseInt(process.env.SMTP_PORT || '465', 10);
const APP_NAME = 'Qwen Chat';

const transporter =
  SMTP_USER && SMTP_PASS
    ? nodemailer.createTransport({
        host: process.env.SMTP_HOST || 'smtp.gmail.com',
        port: SMTP_PORT,
        secure: SMTP_PORT === 465,
        auth: { user: SMTP_USER, pass: SMTP_PASS },
      })
    : null;

const COPY = {
  verify_email: {
    subject: `Verify your email for ${APP_NAME}`,
    intro: 'Use this code to verify your email and finish creating your account.',
  },
  change_password: {
    subject: `Confirm your ${APP_NAME} password change`,
    intro: 'Use this code to confirm changing your password.',
  },
  change_email: {
    subject: `Confirm your new ${APP_NAME} email`,
    intro: 'Use this code to confirm this address as your new account email.',
  },
};

async function sendCodeEmail(to, code, purpose, ttlMinutes) {
  const copy = COPY[purpose];
  if (!copy) throw new Error(`Unknown email purpose: ${purpose}`);

  if (!transporter) {
    if (process.env.NODE_ENV === 'production') {
      throw new Error('SMTP is not configured (set SMTP_USER and SMTP_PASS)');
    }
    console.log(`[mail:dev] ${purpose} code for ${to}: ${code}`);
    return;
  }

  const text =
    `${copy.intro}\n\nYour code: ${code}\n\n` +
    `It expires in ${ttlMinutes} minutes. If you didn't ask for this, you can ignore this email.`;

  const html =
    `<div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:420px">` +
    `<p>${copy.intro}</p>` +
    `<p style="font-size:32px;font-weight:700;letter-spacing:6px;margin:16px 0">${code}</p>` +
    `<p style="color:#666">It expires in ${ttlMinutes} minutes. If you didn't ask for this, you can ignore this email.</p>` +
    `</div>`;

  await transporter.sendMail({
    from: process.env.MAIL_FROM || `"${APP_NAME}" <${SMTP_USER}>`,
    to,
    subject: copy.subject,
    text,
    html,
  });
}

module.exports = { sendCodeEmail };