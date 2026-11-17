#!/usr/bin/env python3
"""OpenAI-compatible /v1/chat/completions in front of a logged-in agent CLI on the host, for detector:=vlm.

claude-* models go to the Claude Code CLI, gpt-* models to the Codex CLI (both logged in with
the subscription). Runs on the host (where the CLIs are logged in), not in the container. With WSL mirrored
networking the container reaches it on localhost:
    python scripts/cli_vlm_server.py            # port 11436
    ros2 launch vs_bringup servo.launch.py detector:=vlm  (VS_VLM_URL=http://localhost:11436/v1,
                                                           VS_VLM_MODEL=claude-opus-5-5)
Each request is one non-interactive CLI call (several seconds, counts against the subscription), so only
use it with a long detector_period.
"""
import argparse
import base64
import json
import subprocess
import os
import shutil
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

MODELS = ['claude-opus-5-5', 'claude-sonnet-5-5', 'claude-haiku-5-5', 'gpt-6-astra',
          'gpt-6.1-terra']
# Codex: medium effort, comparable to the Claude calls (default effort), not the top setting
CODEX_EFFORT = os.environ.get('VLM_CODEX_EFFORT', 'medium')
CLI = os.environ.get('VLM_CLI', 'claude')
SYSTEM = 'You are an object detector. Answer only with the requested JSON, no prose.'


def ask_codex(model, text, image_bytes, timeout):
    with tempfile.TemporaryDirectory() as tmp:
        img, answer = Path(tmp) / 'image.jpg', Path(tmp) / 'answer.txt'
        img.write_bytes(image_bytes)
        # read-only sandbox: the model only looks at the attached image. stdin must be empty:
        # started from WSL the server inherits a pipe, and codex exec waits for its end
        cmd = [shutil.which('codex') or 'codex', 'exec', '--skip-git-repo-check', '-s', 'read-only',
               '-m', model, '-c', f'model_reasoning_effort={CODEX_EFFORT}', '-i', str(img),
               '-o', str(answer), f'{SYSTEM} {text}']
        out = subprocess.run(cmd, cwd=tmp, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                             timeout=timeout,
                             encoding='utf-8')
        if out.returncode != 0 or not answer.exists():
            raise RuntimeError(out.stderr.strip()[-500:] or 'codex gave no answer')
        return answer.read_text(encoding='utf-8')


def ask_cli(model, text, image_bytes, timeout):
    if model.startswith('gpt'):
        return ask_codex(model, text, image_bytes, timeout)
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / 'image.jpg').write_bytes(image_bytes)
        # --bare would skip the subscription login (API key only); an empty setting source list
        # and no MCP servers keep the call small instead
        cmd = [CLI, '-p', '--model', model, '--output-format', 'json',
               '--tools', 'Read', '--allowedTools', 'Read', '--no-session-persistence',
               '--strict-mcp-config', '--setting-sources', '', '--system-prompt', SYSTEM,
               f'Read the image file image.jpg. {text}']
        out = subprocess.run(cmd, cwd=tmp, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                             timeout=timeout,
                             shell=False, encoding='utf-8')
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip() or out.stdout.strip())
    return json.loads(out.stdout)['result']


class Handler(BaseHTTPRequestHandler):
    timeout_s = 120
    last_request = time.time()

    def _send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        Handler.last_request = time.time()
        if self.path.rstrip('/').endswith('/models'):
            self._send(200, {'object': 'list', 'data': [{'id': m, 'object': 'model'}
                                                        for m in MODELS]})
        else:
            self._send(404, {'error': 'not found'})

    def do_POST(self):
        if not self.path.rstrip('/').endswith('/chat/completions'):
            return self._send(404, {'error': 'not found'})
        Handler.last_request = time.time()
        req = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        texts, image = [], None
        for part in req['messages'][-1]['content']:
            if part['type'] == 'text':
                texts.append(part['text'])
            elif part['type'] == 'image_url':
                image = base64.b64decode(part['image_url']['url'].split(',', 1)[1])
        if image is None:
            return self._send(400, {'error': 'no image'})
        t0 = time.time()
        try:
            answer = ask_cli(req.get('model', MODELS[0]), ' '.join(texts), image,
                                self.timeout_s)
        except Exception as exc:  # report to the detector instead of dropping the connection
            return self._send(502, {'error': str(exc)})
        self.log_message('%.1f s: %s', time.time() - t0, answer.replace('\n', ' ')[:200])
        self._send(200, {'object': 'chat.completion', 'model': req.get('model'),
                         'choices': [{'index': 0, 'finish_reason': 'stop',
                                      'message': {'role': 'assistant', 'content': answer}}]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=11436)
    ap.add_argument('--timeout', type=float, default=120)
    ap.add_argument('--idle-exit', type=float, help='quit after this many seconds without requests')
    args = ap.parse_args()
    Handler.timeout_s = args.timeout
    if args.idle_exit:
        def watchdog():
            while time.time() - Handler.last_request < args.idle_exit:
                time.sleep(10)
            os._exit(0)
        threading.Thread(target=watchdog, daemon=True).start()
    # Single-threaded on purpose: parallel CLI calls only burn quota
    HTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
