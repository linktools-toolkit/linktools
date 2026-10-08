#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optional native nginx request tests; mock WAF is not the SafeLine release gate."""
import base64
import contextlib
import hashlib
import http.client
import http.server
import json
import os
import shutil
import socket
import ssl
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest

from _harness import builtin_consumer_type


NginxGeneration = builtin_consumer_type("100-nginx")


def _port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_native_waf_auth_metadata_and_credential_headers(fresh_manager, tmp_path):
    binary = os.environ.get("CNTR_TEST_NGINX") or shutil.which("nginx")
    if not binary or not shutil.which("openssl"):
        pytest.skip("Native nginx with SSL, realip, auth_request and openssl is required")
    nginx = fresh_manager.containers["nginx"]
    events = []
    release_stream = threading.Event()
    origin_port = _port()

    class Backend(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.handle_request()

        def do_POST(self):
            self.handle_request()

        def handle_request(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            kind = self.server.kind
            events.append((kind, self.path, self.command, dict(self.headers), body))
            if kind == "waf":
                connection = http.client.HTTPConnection(
                    "127.0.0.253", origin_port, source_address=("127.0.0.254", 0), timeout=3)
                connection.request(self.command, self.path, body, dict(self.headers))
                response = connection.getresponse()
                data = response.read()
                self.send_response(response.status)
                for key, value in response.getheaders():
                    if key.lower() not in ("connection", "transfer-encoding", "content-length"):
                        self.send_header(key, value)
                self.end_headers()
                self.wfile.write(data)
                connection.close()
                return
            if kind == "app" and self.path == "/waf-public/sse":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b"data: first\n\n")
                self.wfile.flush()
                release_stream.wait(2)
                self.wfile.write(b"data: second\n\n")
                return
            if kind == "app" and self.path == "/waf-public/ws":
                self.protocol_version = "HTTP/1.1"
                self.send_response(101)
                self.send_header("Upgrade", "websocket")
                self.send_header("Connection", "Upgrade")
                accept = base64.b64encode(hashlib.sha1((self.headers["Sec-WebSocket-Key"] +
                    "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()).decode()
                self.send_header("Sec-WebSocket-Accept", accept)
                self.end_headers()
                frame = self.rfile.read(2)
                mask = self.rfile.read(4)
                payload = self.rfile.read(frame[1] & 127)
                payload = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
                self.wfile.write(bytes((130, len(payload))) + payload)
                self.wfile.flush()
                self.close_connection = True
                return
            if kind == "auth":
                self.send_response(int(self.headers.get("X-Test-Auth-Status", "204")))
                self.send_header("Remote-User", "verified")
                self.send_header("Location", "https://login.example.test/")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Remote-User", "untrusted-app-user")
            self.end_headers()
            self.wfile.write(json.dumps({"headers": dict(self.headers), "body": body.hex(),
                                        "method": self.command, "path": self.path}).encode())

    with contextlib.ExitStack() as stack:
        def backend(kind, host="127.0.0.1", port=0):
            server = http.server.ThreadingHTTPServer((host, port), Backend)
            server.kind = kind
            stack.callback(server.server_close)
            stack.callback(server.shutdown)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            return server.server_address[1]

        app_port = backend("app")
        auth_port = backend("auth")
        backend("waf", "127.0.0.254", origin_port)
        http_port, https_port, health_port = _port(), _port(), _port()
        config = {"NGINX_HTTP_PORT": http_port, "NGINX_HTTPS_PORT": https_port,
                  "NGINX_WAF_PORT": origin_port, "NGINX_ROOT_DOMAIN": "test",
                  "SAFELINE_SUBNET_PREFIX": "127.0.0"}
        producer = SimpleNamespace(name="native-fixture", env_config=config)
        site = SimpleNamespace(server_name="app.test", file_id="native", var_name="native",
                               local_id="web", default=False, https=True, waf=True, auth=True,
                               waf_bypass=(r"^/waf-public", r"^/both"),
                               auth_bypass=(r"^/auth-public", r"^/both"),
                               auth_headers={"Authorization": "Bearer secret$host"}, vars={},
                               proxy="http://127.0.0.1:" + str(app_port))

        def render(name, selected=site):
            return NginxGeneration(nginx).render_template(producer, nginx.get_source_path("templates", name), selected)

        root = render("nginx.conf", SimpleNamespace(vars={"generation_id": "native-test", "waf": True, "site_files": ("sites/native.conf",)}))
        # The fixture changes only sandbox resources and loopback endpoint addresses.
        root = root.replace("include /etc/nginx/mime.types;", "")
        root = root.replace("/var/log/nginx/error.log", str(tmp_path / "error.log"))
        root = root.replace("/var/log/nginx/access.log", str(tmp_path / "access.log"))
        root = root.replace("/var/run/nginx.pid", str(tmp_path / "nginx.pid"))
        root = root.replace("worker_processes auto", "worker_processes 1")
        if os.environ.get("CNTR_NGINX_TEST_TCP_HEALTH") == "1":
            # This opt-in never establishes the production Unix-socket health gate.
            root = root.replace("unix:/run/nginx-health.sock", "127.0.0.1:" + str(health_port))
        else:
            root = root.replace("/run/nginx-health.sock", str(tmp_path / "health.sock"))
        (tmp_path / "logs").mkdir()
        (tmp_path / "sites").mkdir()
        (tmp_path / "nginx.conf").write_text(root)
        header_text = "\n".join("proxy_set_header " + nginx.complex_value(key) + " " + value + ";"
                                for key, value in nginx.header_items(site))
        proxy = "set $target http://127.0.0.1:" + str(app_port) + "; proxy_pass $target;"
        business = render("default.conf") + "\n" + "\n".join((
            "location /off { auth_request off; " + header_text + proxy + " }",
            "location /fallback { error_page 403 = /off; " + header_text + proxy + " }",
            "location ~ ^/capture/(.+)$ { " + header_text + "proxy_set_header X-Capture $1; " + proxy + " }",
        ))
        server_text = NginxGeneration(nginx).render_template(
            producer, nginx.get_source_path("templates", "server.conf"), site, business=business)
        server_text = server_text.replace("/etc/certs/", str(tmp_path) + "/")
        server_text = server_text.replace("listen " + str(origin_port) + " ",
                                          "listen 127.0.0.253:" + str(origin_port) + " ")
        server_text = server_text.replace("http://authelia:9091", "http://127.0.0.1:" + str(auth_port))
        (tmp_path / "sites/native.conf").write_text(server_text)
        subprocess.check_call(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
                               "-subj", "/CN=app.test", "-keyout", str(tmp_path / "test_key.pem"),
                               "-out", str(tmp_path / "test_fullchain.pem"), "-days", "1"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        command = [binary, "-p", str(tmp_path) + "/", "-c", str(tmp_path / "nginx.conf")]
        validation = subprocess.run(command + ["-t"], check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if b"Operation not permitted" in validation.stdout:
            pytest.skip("Sandbox cannot create the configured native nginx socket")
        assert validation.returncode == 0, validation.stdout.decode()
        process = subprocess.Popen(command + ["-g", "daemon off; master_process off;"])
        stack.callback(process.wait, timeout=5)
        stack.callback(process.terminate)
        deadline = time.monotonic() + 5
        while True:
            try:
                with socket.create_connection(("127.0.0.1", https_port), timeout=.1):
                    break
            except OSError:
                assert time.monotonic() < deadline, "nginx did not begin listening"
                time.sleep(.02)

        def request(path, headers=None):
            request_headers = {"Host": "app.test:" + str(https_port), "Authorization": "Bearer client",
                               "X-Auth-User": "forged", "X-Proxy-Original-Scheme": "forged",
                               "X-Forwarded-For": "forged", "Forwarded": "forged"}
            request_headers.update(headers or {})
            connection = http.client.HTTPSConnection("127.0.0.1", https_port, timeout=3,
                                                    context=ssl._create_unverified_context())
            connection.request("POST", path, b"\x00\xffbody", request_headers)
            response = connection.getresponse()
            result = response.status, response.read()
            connection.close()
            return result

        for path, waf, auth in (("/private?x=%2f", True, True), ("/auth-public/a", True, False),
                                ("/waf-public/a", False, True), ("/both/a", False, False),
                                ("/capture/hello", True, True), ("/off/a", True, False)):
            events.clear()
            status, data = request(path)
            assert status == 200, data
            response = json.loads(data)
            assert response["method"] == "POST" and response["path"] == path
            assert response["body"] == "00ff626f6479"
            assert [event[0] for event in events] == (["waf"] if waf else []) + (["auth"] if auth else []) + ["app"]
            headers = response["headers"]
            assert headers["Authorization"] == ("Bearer secret$host" if auth else "Bearer client")
            assert headers.get("X-Auth-User") == ("verified" if auth else None)
            assert headers["X-Original-URL"] == "https://app.test:" + str(https_port) + path
            assert headers["X-Original-Method"] == "POST"
            assert headers["X-Forwarded-For"] == "127.0.0.1"
            assert headers["X-Forwarded-Port"] == str(https_port)
            assert "Forwarded" not in headers
            assert not any(key.startswith("X-Proxy-Original-") for key in headers)
            for kind, _, _, headers, _ in events:
                if kind == "auth":
                    assert not any(key.startswith(("X-Proxy-Original-", "X-Auth-")) for key in headers)
                    assert headers["X-Original-Method"] == "POST"
            if path.startswith("/capture"):
                assert response["headers"]["X-Capture"] == "hello"

        for denied in (401, 403, 500):
            events.clear()
            status, _ = request("/private", {"X-Test-Auth-Status": str(denied)})
            assert status == (302 if denied == 401 else denied)
            assert "app" not in [event[0] for event in events]
        status, data = request("/fallback", {"X-Test-Auth-Status": "403"})
        assert status == 200 and json.loads(data)["headers"]["Authorization"] == "Bearer client"
        assert "X-Auth-User" not in json.loads(data)["headers"]
        # Stream bytes must arrive before the backend completes the response.
        connection = http.client.HTTPSConnection("127.0.0.1", https_port, timeout=1,
                                                context=ssl._create_unverified_context())
        connection.request("GET", "/waf-public/sse", headers={"Host": "app.test"})
        response = connection.getresponse()
        assert response.status == 200
        try:
            assert response.readline() == b"data: first\n"
        finally:
            release_stream.set()
        assert b"data: second" in response.read()
        connection.close()
        # Exercise an actual upgraded, bidirectional WebSocket binary frame.
        tls = ssl._create_unverified_context().wrap_socket(
            socket.create_connection(("127.0.0.1", https_port), timeout=2), server_hostname="app.test")
        try:
            tls.sendall(b"GET /waf-public/ws HTTP/1.1\r\nHost: app.test\r\nUpgrade: websocket\r\n"
                        b"Connection: Upgrade\r\nSec-WebSocket-Version: 13\r\n"
                        b"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\r\n")
            response = b""
            while not response.endswith(b"\r\n\r\n"):
                chunk = tls.recv(1)
                assert chunk
                response += chunk
            assert response.startswith(b"HTTP/1.1 101")
            assert b"s3pPLMBiTxaQ9kYGzzhZRbK+xOo=" in response
            mask, payload = b"mask", b"ping"
            tls.sendall(bytes((130, 128 + len(payload))) + mask +
                        bytes(value ^ mask[index % 4] for index, value in enumerate(payload)))
            frame = b""
            while len(frame) < 6:
                chunk = tls.recv(6 - len(frame))
                assert chunk
                frame += chunk
            assert frame == b"\x82\x04ping"
        finally:
            tls.close()
        # The origin accepts only the actual WAF socket and matching original metadata.
        connection = http.client.HTTPConnection("127.0.0.253", origin_port, timeout=3)
        connection.request("GET", "/private", headers={"Host": "app.test"})
        assert connection.getresponse().status == 403
        connection.close()
        for headers in ({"Host": "app.test"}, {"Host": "app.test", "X-Proxy-Original-Scheme": "https",
                        "X-Proxy-Original-Host": "other.test", "X-Proxy-Original-URI": "/private",
                        "X-Proxy-Original-Method": "GET", "X-Proxy-Original-Client-IP": "127.0.0.1"}):
            connection = http.client.HTTPConnection("127.0.0.253", origin_port, timeout=3,
                                                    source_address=("127.0.0.254", 0))
            connection.request("GET", "/private", headers=headers)
            assert connection.getresponse().status == 400
            connection.close()
