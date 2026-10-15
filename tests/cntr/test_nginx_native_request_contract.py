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

from linktools.core import Config, ConfigSchema, DictSource


def _native_site(port):
    return SimpleNamespace(server_name="app.test", file_id="native", var_name="native",
                           key="web", default_server=False, https=True, waf=True, auth=True,
                           waf_bypass=(r"^/waf-public", r"^/both"),
                           auth_bypass=(r"^/auth-public", r"^/both"),
                           auth_headers={"Authorization": "Bearer secret$host"}, template_vars={},
                           proxy="http://127.0.0.1:" + str(port))


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
    auth_responses = []
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
            if kind == "app" and self.path == "/waf-public/sse":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b"data: first\n\n")
                self.wfile.flush()
                release_stream.wait(2)
                self.wfile.write(b"data: second\n\n")
                return
            if kind == "app" and self.path.endswith("/ws"):
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
                status = auth_responses.pop(0) if auth_responses else int(self.headers.get("X-Test-Auth-Status", "204"))
                self.send_response(status)
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
        http_port, https_port, health_port = _port(), _port(), _port()
        config = {"NGINX_HTTP_PORT": http_port, "NGINX_HTTPS_PORT": https_port,
                  "NGINX_WAF_PORT": origin_port, "NGINX_ROOT_DOMAIN": "test",
                  "SAFELINE_SUBNET_PREFIX": "127.0.0"}
        producer = SimpleNamespace(name="native-fixture", env_config=Config(None, ConfigSchema(), [DictSource(config)]))
        site = _native_site(app_port)

        def render(name, selected=site):
            return nginx._render_site_template(producer, nginx.get_source_path("templates", name), selected)

        root = render("nginx.conf", SimpleNamespace(template_vars={"generation_id": "native-test", "waf": True, "site_files": ("sites/native.conf",)}))
        # The fixture changes only sandbox resources and loopback endpoint addresses.
        root = root.replace("include /etc/nginx/mime.types;", "")
        root = root.replace("/etc/nginx/managed/", str(tmp_path) + "/")
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
        # Model the verified SafeLine XFF append contract with a native proxy,
        # including WebSocket upgrades and the two distinct trusted proxy sockets.
        root = root.rsplit("}", 1)[0] + """
        server {
            listen 127.0.0.254:PORT;
            location / {
                proxy_bind 127.0.0.254;
                proxy_http_version 1.1;
                proxy_set_header Host $http_host;
                proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
                proxy_set_header X-Fixture-Waf-Input $http_x_forwarded_for;
                proxy_set_header Upgrade $http_upgrade;
                proxy_set_header Connection $connection_upgrade;
                proxy_pass http://127.0.0.253:PORT;
            }
        }
        }
        """.replace("PORT", str(origin_port))
        (tmp_path / "nginx.conf").write_text(root)
        header_text = "\n".join("proxy_set_header " + nginx.complex_value(key) + " " + value + ";"
                                for key, value in nginx.header_items(site))
        proxy = "set $target http://127.0.0.1:" + str(app_port) + "; proxy_pass $target;"
        business = render("default.conf") + "\n" + "\n".join((
            "location /off { auth_request off; " + header_text + proxy + " }",
            "location /fallback { error_page 403 = /off; " + header_text + proxy + " }",
            "location /reauth { error_page 403 = /capture/hello; " + header_text + proxy + " }",
            "location ~ ^/capture/(.+)$ { " + header_text + "proxy_set_header X-Capture $1; " + proxy + " }",
            "location ~ ^/rewrite/(.+)$ { set $target http://127.0.0.1:" + str(app_port) +
            "; rewrite ^/rewrite/(.+)$ /$1 break; " + header_text + "proxy_pass $target; }",
        ))
        server_text = nginx._render_site_template(
            producer, nginx.get_source_path("templates", "server.conf"), site, business=business)
        plain_site = _native_site(app_port)
        plain_site.server_name = "plain.test"
        plain_site.var_name = "plain"
        plain_site.https = False
        plain_site.auth = False
        server_text += nginx._render_site_template(
            producer, nginx.get_source_path("templates", "server.conf"), plain_site,
            business=render("default.conf", plain_site))
        server_text = server_text.replace("location @waf {", "location @waf { proxy_bind 127.0.0.253;")
        server_text = server_text.replace("/etc/certs/" + nginx.cert_image_revision + "/live/", str(tmp_path) + "/")
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
            assert [event[0] for event in events] == (["auth"] if auth else []) + ["app"]
            headers = response["headers"]
            assert headers["Authorization"] == ("Bearer secret$host" if auth else "Bearer client")
            assert headers.get("X-Auth-User") == ("verified" if auth else None)
            assert headers["X-Original-URL"] == "https://app.test:" + str(https_port) + path
            assert headers["X-Original-Method"] == "POST"
            assert headers.get("X-Fixture-Waf-Input") == ("127.0.0.1" if waf else None)
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

        # The subrequest-only status regex must leave business captures intact.
        for accepted in (200, 204, 299):
            status, data = request("/capture/hello", {"X-Test-Auth-Status": str(accepted)})
            assert status == 200, data
            headers = json.loads(data)["headers"]
            assert headers["X-Capture"] == "hello"
            assert headers["X-Auth-User"] == "verified"
            assert headers["Authorization"] == "Bearer secret$host"

        # Internal redirects can authenticate twice with shared variable storage.
        for responses, expected in (((403, 204), 200), ((403, 401), 401)):
            events.clear()
            auth_responses[:] = responses
            status, data = request("/reauth")
            assert status == expected, data
            assert not auth_responses
            if expected == 200:
                headers = json.loads(data)["headers"]
                assert headers["X-Capture"] == "hello"
                assert headers["X-Auth-User"] == "verified"
                assert headers["Authorization"] == "Bearer secret$host"
            else:
                assert "app" not in [event[0] for event in events]

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
        for path in ("/auth-public/ws", "/waf-public/ws"):
            tls = ssl._create_unverified_context().wrap_socket(
                socket.create_connection(("127.0.0.1", https_port), timeout=2), server_hostname="app.test")
            try:
                tls.sendall(("GET " + path + " HTTP/1.1\r\nHost: app.test\r\nUpgrade: websocket\r\n"
                            "Connection: Upgrade\r\nSec-WebSocket-Version: 13\r\n"
                            "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\r\n").encode())
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
        events.clear()
        status, data = request("/rewrite/rewritten?x=%2f")
        assert status == 200
        payload = json.loads(data)
        assert payload["path"] == "/rewritten?x=%2f"
        assert payload["method"] == "POST"
        assert payload["headers"]["X-Original-Method"] == "POST"
        assert payload["headers"]["X-Original-URL"] == "https://app.test:" + str(https_port) + "/rewrite/rewritten?x=%2f"
        assert events[0][0] == "auth"
        assert events[0][3]["X-Original-URL"] == payload["headers"]["X-Original-URL"]
        assert events[0][3]["X-Original-Method"] == "POST"

        events.clear()
        connection = http.client.HTTPConnection("127.0.0.1", http_port, timeout=3)
        connection.request("GET", "/private", headers={"Host": "app.test"})
        response = connection.getresponse()
        assert response.status == 301
        response.read()
        connection.close()
        assert events == []
        connection = http.client.HTTPConnection("127.0.0.1", http_port, timeout=3)
        connection.request("GET", "/plain?x=%2f", headers={"Host": "plain.test", "X-Forwarded-For": "forged"})
        response = connection.getresponse()
        assert response.status == 200
        payload = json.loads(response.read())
        connection.close()
        assert payload["headers"]["X-Original-URL"] == "http://plain.test/plain?x=%2f"
        assert payload["headers"]["X-Forwarded-For"] == "127.0.0.1"

        # Origin admission is independent of client identity or custom headers.
        connection = http.client.HTTPConnection("127.0.0.253", origin_port, timeout=3)
        connection.request("GET", "/private", headers={"Host": "app.test"})
        assert connection.getresponse().status == 403
        connection.close()

        def origin_request(path, headers, source="127.0.0.254", destination="127.0.0.253"):
            connection = http.client.HTTPConnection(destination, origin_port, timeout=3,
                                                    source_address=(source, 0))
            connection.request("GET", path, headers=dict({"Host": "app.test"}, **headers))
            response = connection.getresponse()
            result = response.status, response.read()
            connection.close()
            return result

        assert origin_request("/private", {"X-Forwarded-For": "198.51.100.1"}, "127.0.0.253")[0] == 403
        events.clear()
        status, data = origin_request("/probe", {"Host": "plain.test"})
        assert status == 200
        assert [event[0] for event in events] == ["app"]
        assert "X-Forwarded-For" not in json.loads(data)["headers"]

        events.clear()
        status, data = origin_request("/auth-public/probe?x=%2f", {}, "127.0.0.4", "127.0.0.254")
        assert status == 200
        assert json.loads(data)["headers"]["X-Forwarded-For"] == "127.0.0.4"
        assert [event[0] for event in events] == ["app"]

        for headers, expected_ip in (
            ({}, None),
            ({"X-Forwarded-For": "2001:0db8:0:0::1, 127.0.0.253"}, "2001:db8::1"),
            ({"X-Forwarded-For": "invalid"}, None),
            ({"X-Proxy-Original-Host": "other.test", "X-Proxy-Original-Scheme": "invalid"}, None),
        ):
            status, data = origin_request("/auth-public/probe?x=%2f", headers)
            assert status == 200
            forwarded = json.loads(data)["headers"]
            assert forwarded.get("X-Forwarded-For") == expected_ip
            assert forwarded.get("X-Real-IP") == expected_ip
            assert forwarded["X-Original-URL"] == "https://app.test/auth-public/probe?x=%2f"
            assert forwarded["X-Original-Method"] == "GET"

        events.clear()
        auth_responses[:] = [401]
        status, _ = origin_request("/private", {})
        assert status == 302
        assert [event[0] for event in events] == ["auth"]
        assert "X-Forwarded-For" not in events[0][3]
        assert "X-Real-IP" not in events[0][3]



def test_native_fixture_template_contract_without_native_processes(fresh_manager):
    nginx = fresh_manager.containers["nginx"]
    producer = SimpleNamespace(name="native-fixture", env_config=Config(None, ConfigSchema(), [DictSource({
        "NGINX_HTTP_PORT": 8080, "NGINX_HTTPS_PORT": 8443,
        "NGINX_WAF_PORT": 8081, "NGINX_ROOT_DOMAIN": "test",
        "SAFELINE_SUBNET_PREFIX": "127.0.0",
    })]))
    site = _native_site(8082)
    business = nginx._render_site_template(producer, nginx.get_source_path("templates", "default.conf"), site)
    server = nginx._render_site_template(
        producer, nginx.get_source_path("templates", "server.conf"), site, business=business)
    root = nginx._render_site_template(producer, nginx.get_source_path("templates", "nginx.conf"),
        SimpleNamespace(template_vars={"generation_id": "native-test", "waf": True, "site_files": ("sites/native.conf",)}))
    assert "app.test" in server
    assert "sites/native.conf" in root
    assert "http://127.0.0.1:8082" in business



def test_native_shared_host_keeps_public_and_authenticated_routes_isolated(fresh_manager, tmp_path):
    binary = os.environ.get("CNTR_TEST_NGINX") or shutil.which("nginx")
    if not binary or not shutil.which("openssl"):
        pytest.skip("Native nginx and openssl are required")

    nginx = fresh_manager.containers["nginx"]

    class Backend(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.server.paths.append(self.path)
            self.server.requests.append(dict(self.headers))
            if self.server.kind == "auth":
                status = getattr(self.server, "auth_status", 204)
                self.send_response(status)
                if status == 204:
                    self.send_header("Remote-User", "verified")
                else:
                    self.send_header("Location", "https://login.example.test/")
                self.end_headers()
                return
            payload = json.dumps({
                "path": self.path, "user": self.headers.get("X-Auth-User"),
                "authorization": self.headers.get("Authorization"),
            }).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(payload)

    with contextlib.ExitStack() as stack:
        def start_backend(kind):
            server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Backend)
            server.kind = kind
            server.paths = []
            server.requests = []
            stack.callback(server.server_close)
            stack.callback(server.shutdown)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            return server

        app = start_backend("app")
        auth = start_backend("auth")
        http_port, https_port = _port(), _port()
        config = {
            "NGINX_HTTP_PORT": http_port, "NGINX_HTTPS_PORT": https_port,
            "NGINX_ROOT_DOMAIN": "test",
        }
        producer = SimpleNamespace(name="shared-native", env_config=Config(None, ConfigSchema(), [DictSource(config)]))

        def site(name, auth_enabled):
            return SimpleNamespace(
                producer=producer, key=name, file_id=name, var_name=name,
                server_name="app.test", default_server=False, https=True, waf=False,
                auth=auth_enabled, auth_bypass=(r"^/admin/free",) if auth_enabled else (),
                waf_bypass=(), auth_headers={"Authorization": "Bearer trusted"} if auth_enabled else {},
                template_vars={},
            )

        public = site("public", False)
        admin = site("admin", True)
        business = []
        for member, location in ((public, "/public"), (admin, "/admin")):
            path = tmp_path / (member.key + ".j2")
            path.write_text(
                '{% from "nginx/headers.j2" import proxy_headers, route_authorization with context %}'
                'location ' + location + ' { {{ route_authorization() }} {{ proxy_headers() }} '
                'proxy_pass http://127.0.0.1:' + str(app.server_port) + '; }')
            business.append(nginx._render_site_template(
                producer, path, member, route_auth=True))
        server = nginx._render_site_template(
            producer, nginx.get_source_path("templates", "server.conf"), public,
            business="\n".join(business), sites=(public, admin), route_auth=True)
        server = server.replace("/etc/certs/" + nginx.cert_image_revision + "/live/", str(tmp_path) + "/")
        server = server.replace("http://authelia:9091", "http://127.0.0.1:" + str(auth.server_port))

        root = nginx._render_site_template(
            producer, nginx.get_source_path("templates", "nginx.conf"),
            SimpleNamespace(template_vars={"generation_id": "native", "waf": False,
                                  "site_files": ("sites/shared.conf",)}))
        root = root.replace("include /etc/nginx/mime.types;", "")
        root = root.replace("/etc/nginx/managed/", str(tmp_path) + "/")
        root = root.replace("/var/log/nginx/error.log", str(tmp_path / "error.log"))
        root = root.replace("/var/log/nginx/access.log", str(tmp_path / "access.log"))
        root = root.replace("/var/run/nginx.pid", str(tmp_path / "nginx.pid"))
        if os.environ.get("CNTR_NGINX_TEST_TCP_HEALTH") == "1":
            root = root.replace("unix:/run/nginx-health.sock", "127.0.0.1:" + str(_port()))
        else:
            root = root.replace("/run/nginx-health.sock", str(tmp_path / "health.sock"))
        root = root.replace("worker_processes auto", "worker_processes 1")
        (tmp_path / "sites").mkdir()
        (tmp_path / "sites/shared.conf").write_text(server)
        (tmp_path / "nginx.conf").write_text(root)
        subprocess.check_call([
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
            "-subj", "/CN=app.test", "-keyout", str(tmp_path / "test_key.pem"),
            "-out", str(tmp_path / "test_fullchain.pem"), "-days", "1"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        command = [binary, "-p", str(tmp_path) + "/", "-c", str(tmp_path / "nginx.conf")]
        check = subprocess.run(command + ["-t"], capture_output=True, text=True)
        if "Operation not permitted" in check.stderr:
            pytest.skip("Native nginx listener is unavailable in this environment")
        assert check.returncode == 0, check.stderr
        process = subprocess.Popen(command + ["-g", "daemon off; master_process off;"])
        stack.callback(process.wait, timeout=5)
        stack.callback(process.terminate)
        for _ in range(100):
            try:
                with socket.create_connection(("127.0.0.1", https_port), timeout=0.1):
                    break
            except OSError:
                time.sleep(0.05)
        else:
            pytest.fail("Native nginx did not start")

        def request(path):
            connection = http.client.HTTPSConnection(
                "127.0.0.1", https_port, timeout=3,
                context=ssl._create_unverified_context())
            connection.request("GET", path, headers={
                "Host": "app.test", "Authorization": "Bearer client", "X-Forwarded-For": "198.51.100.99"})
            response = connection.getresponse()
            result = response.status, json.loads(response.read())
            connection.close()
            return result

        assert request("/public") == (200, {
            "path": "/public", "user": None, "authorization": "Bearer client"})
        assert request("/admin") == (200, {
            "path": "/admin", "user": "verified", "authorization": "Bearer trusted"})
        assert request("/admin/free") == (200, {
            "path": "/admin/free", "user": None, "authorization": "Bearer client"})
        assert auth.paths == ["/api/authz/auth-request"]
        assert all(headers["X-Forwarded-For"] == "127.0.0.1"
                   for headers in app.requests + auth.requests)
        app_paths = list(app.paths)
        auth.auth_status = 401
        connection = http.client.HTTPSConnection(
            "127.0.0.1", https_port, timeout=3,
            context=ssl._create_unverified_context())
        connection.request("GET", "/admin", headers={"Host": "app.test"})
        denied = connection.getresponse()
        assert denied.status == 302
        assert denied.getheader("Location") == "https://login.example.test/"
        denied.read()
        connection.close()
        assert app.paths == app_paths


@pytest.mark.parametrize("waf", [False, True])
@pytest.mark.parametrize("limit,maximum", [(None, 1024 * 1024), ("10m", 10 * 1024 * 1024), (0, None)])
def test_native_body_limits_with_fixed_length_and_chunked_requests(fresh_manager, tmp_path, waf, limit, maximum):
    binary = os.environ.get("CNTR_TEST_NGINX") or shutil.which("nginx")
    if not binary:
        pytest.skip("Native nginx with realip is required")
    nginx = fresh_manager.containers["nginx"]
    received = []

    class Backend(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            size = 0
            if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
                while True:
                    chunk = int(self.rfile.readline().split(b";", 1)[0], 16)
                    if not chunk:
                        self.rfile.readline()
                        break
                    size += len(self.rfile.read(chunk))
                    self.rfile.read(2)
            else:
                size = len(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            received.append((size, self.headers.get("X-Fixture-Waf")))
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

    with contextlib.ExitStack() as stack:
        app = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Backend)
        stack.callback(app.server_close)
        stack.callback(app.shutdown)
        threading.Thread(target=app.serve_forever, daemon=True).start()
        public_port, origin_port = _port(), _port()
        config = {"NGINX_HTTP_PORT": public_port, "NGINX_WAF_PORT": origin_port,
                  "SAFELINE_SUBNET_PREFIX": "127.0.0"}
        producer = SimpleNamespace(name="body-fixture", env_config=Config(None, ConfigSchema(), [DictSource(config)]))
        site = _native_site(app.server_port)
        site.https, site.waf, site.auth = False, waf, False
        site.waf_bypass = (r"^/bypass",)
        site.template_vars = {} if limit is None else {"client_max_body_size": limit}
        business = nginx._render_site_template(
            producer, nginx.get_source_path("templates", "default.conf"), site)
        # Explicit native limits must override the server policy on both hops.
        business += "\nlocation /small { client_max_body_size 1m; proxy_pass " + site.proxy + "; }"
        business += "\nlocation /bypass/unlimited { client_max_body_size 0; proxy_pass " + site.proxy + "; }"
        server = nginx._render_site_template(
            producer, nginx.get_source_path("templates", "server.conf"), site, business=business)
        server = server.replace("listen " + str(origin_port) + " ",
                                "listen 127.0.0.253:" + str(origin_port) + " ")
        if waf:
            # This is a transport-only WAF fixture, not a SafeLine acceptance test.
            server += """
server {
    listen 127.0.0.254:PORT;
    client_max_body_size 0;
    location / {
        proxy_bind 127.0.0.254;
        proxy_http_version 1.1;
        proxy_set_header Host app.test;
        proxy_set_header X-Fixture-Waf 1;
        proxy_pass http://127.0.0.253:PORT;
    }
}
""".replace("PORT", str(origin_port))
        root = nginx._render_site_template(producer, nginx.get_source_path("templates", "nginx.conf"),
            SimpleNamespace(template_vars={"waf": waf, "site_files": ()}))
        root = root.replace("include /etc/nginx/mime.types;", "")
        # Packaged nginx may use a root-owned absolute path despite the test prefix.
        root = root.replace("http {", "http {\n    client_body_temp_path " +
                            nginx.quote(str(tmp_path / "client_body_temp")) + ";", 1)
        root = root.replace("worker_processes auto", "worker_processes 1")
        for old, new in (("/var/log/nginx/error.log", "error.log"),
                         ("/var/log/nginx/access.log", "access.log"),
                         ("/var/run/nginx.pid", "nginx.pid"),
                         ("/run/nginx-health.sock", "health.sock")):
            root = root.replace(old, str(tmp_path / new))
        if os.environ.get("CNTR_NGINX_TEST_TCP_HEALTH") == "1":
            root = root.replace("unix:" + str(tmp_path / "health.sock"), "127.0.0.1:" + str(_port()))
        root = root.rsplit("}", 1)[0] + server + "\n}\n"
        (tmp_path / "logs").mkdir()
        (tmp_path / "nginx.conf").write_text(root)
        command = [binary, "-p", str(tmp_path) + "/", "-c", str(tmp_path / "nginx.conf")]
        check = subprocess.run(command + ["-t"], capture_output=True, text=True)
        if "Operation not permitted" in check.stderr:
            pytest.skip("Native nginx listener is unavailable in this environment")
        assert check.returncode == 0, check.stderr
        process = subprocess.Popen(command + ["-g", "daemon off; master_process off;"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        stack.callback(process.wait, timeout=5)
        stack.callback(process.terminate)
        deadline = time.monotonic() + 5
        while True:
            try:
                with socket.create_connection(("127.0.0.1", public_port), timeout=.1):
                    break
            except OSError:
                assert time.monotonic() < deadline, "nginx did not begin listening"
                time.sleep(.02)

        def request(size, chunked, path="/"):
            connection = http.client.HTTPConnection("127.0.0.1", public_port, timeout=10)
            payload = b"a" * size
            body = (payload[index:index + 65536] for index in range(0, size, 65536)) if chunked else payload
            try:
                try:
                    connection.request("POST", path, body, {"Host": "app.test"}, encode_chunked=chunked)
                except BrokenPipeError:
                    pass  # nginx can reject Content-Length before the client finishes writing.
                response = connection.getresponse()
                response.read()
                assert response.status in (200, 413), (response.status, (tmp_path / "error.log").read_text())
                return response.status
            finally:
                connection.close()

        for chunked in (False, True):
            for size in (512 * 1024, 1536 * 1024, 10 * 1024 * 1024 + 1):
                expected = 413 if maximum is not None and size > maximum else 200
                assert request(size, chunked) == expected, (limit, waf, chunked, size)
                if expected == 200:
                    assert received[-1] == (size, "1" if waf else None)
            assert request(1536 * 1024, chunked, "/small") == 413
            bypass_size = 1536 * 1024
            expected = 413 if maximum is not None and bypass_size > maximum else 200
            assert request(bypass_size, chunked, "/bypass/inherited") == expected
            if expected == 200:
                assert received[-1] == (bypass_size, None)
            assert request(bypass_size, chunked, "/bypass/unlimited") == 200
            assert received[-1] == (bypass_size, None)
