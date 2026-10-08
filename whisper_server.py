import json, os, tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from faster_whisper import WhisperModel

LANGUAGE = "en"  # set to None to auto-detect
model = WhisperModel("small", device="cpu", compute_type="int8")

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        size = int(self.headers.get("Content-Length", 0))
        data = self.rfile.read(size)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as f:
            f.write(data)
            path = f.name
        try:
            segments, _ = model.transcribe(path, language=LANGUAGE, beam_size=1, vad_filter=True)
            body = json.dumps({"text": " ".join(s.text.strip() for s in segments).strip()}).encode()
            code = 200
        except Exception as e:
            body = json.dumps({"error": str(e)}).encode()
            code = 500
        finally:
            os.remove(path)
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass

print("Whisper ready on http://127.0.0.1:8001")
HTTPServer(("127.0.0.1", 8001), Handler).serve_forever()