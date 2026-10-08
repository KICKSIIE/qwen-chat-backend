"""
Quick test script: sends a message to your chat backend and prints the reply.

Usage:
    pip install requests
    python test_chat.py
"""

import requests

# --- Config ---
# Same machine as the backend? Use localhost.
# Different machine over the internet? Replace with your ngrok URL, e.g.
# "https://abcd-1234.ngrok-free.app"
BACKEND_URL = "http://localhost:3000"
API_KEY = "change-this-to-a-long-random-string"  # must match your .env

def send_message(session_id: str, message: str) -> str:
    response = requests.post(
        f"{BACKEND_URL}/api/chat/simple",
        headers={"x-api-key": API_KEY},
        json={"sessionId": session_id, "message": message},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["reply"]

if __name__ == "__main__":
    reply = send_message("python-test-1", "hello, what model are you?")
    print("Qwen2.5 says:", reply)
