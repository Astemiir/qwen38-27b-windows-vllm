"""One request against a running server, to prove the endpoint answers.

    python\\python.exe <repo>\\tools\\smoke_test.py [port] [model-name]

Prints the reply and the token usage. A server that is up but has no working kernel still answers, so
pair this with the startup-log check in INSTALL.md.
"""
import json
import sys
import urllib.request

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8010
model = sys.argv[2] if len(sys.argv) > 2 else "qwen3.8-27b"

body = {
    "model": model,
    "messages": [{"role": "user", "content": "Reply with exactly one word: ready"}],
    "max_tokens": 64,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": False},
}
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"},
)
data = json.load(urllib.request.urlopen(req, timeout=600))
msg = data["choices"][0]["message"]
print("content:", (msg.get("content") or "").strip())
if msg.get("reasoning_content"):
    print("reasoning:", msg["reasoning_content"].strip()[:200])
print("usage:", data.get("usage"))
