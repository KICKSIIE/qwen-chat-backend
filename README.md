# Qwen2.5 Chat Backend

Sits between your React Native app and Ollama (running Qwen2.5), so the app
never talks to the model directly.

## Setup

1. Make sure Ollama is running with the model pulled:
   ```
   ollama pull qwen2.5:7b
   ```
2. Copy `.env.example` to `.env` and edit values (especially `API_KEY` and
   `OLLAMA_URL` if Ollama is on a different machine).
3. Install and run:
   ```
   npm install
   npm start
   ```
4. Test it's alive: open `http://localhost:3000/health` in a browser, or:
   ```
   curl http://localhost:3000/health
   ```
5. Test the chat endpoint (simple, non-streaming):
   ```
   curl -X POST http://localhost:3000/api/chat/simple \
     -H "Content-Type: application/json" \
     -H "x-api-key: change-this-to-a-long-random-string" \
     -d "{\"sessionId\": \"test1\", \"message\": \"hello\"}"
   ```

## Connecting the React Native app

See `ChatScreen.example.js` for a full working screen. Key points:
- If testing on a phone against your PC on the same WiFi, use your PC's
  local network IP (not "localhost") as `BACKEND_URL`.
- The `x-api-key` header must match `API_KEY` in your `.env`.
- `sessionId` should be a stable per-user ID so conversation history is kept
  separately per user.

## Moving to a cloud VM later

- Set `OLLAMA_URL` to point at wherever Ollama ends up running (same VM as
  this backend, or a different one).
- Put this server behind HTTPS (Caddy/nginx) before pointing a real app at
  it — never send the raw API key over plain HTTP on a public network.
- Replace the in-memory `conversations` Map with a real database if you need
  history to survive server restarts.
