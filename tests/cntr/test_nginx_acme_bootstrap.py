#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build-time ACME inputs and offline certificate installation with the current script."""

import os
import shutil
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def certificate_case(fresh_manager, tmp_path):
    container = fresh_manager.containers["nginx"]
    settings = fresh_manager.env_config
    settings.set("NGINX_ROOT_DOMAIN", "example.test")
    settings.set("NGINX_HTTPS_ENABLE", True)
    settings.set("ACME_DNS_API", "dns_cf")
    settings.set("CF_Token", "fake-secret")
    container.__dict__["acme_ssl_domains"] = ["example.test", "*.example.test"]
    for name in ("docker_file", "docker_compose"):
        container.__dict__.pop(name, None)

    seed = tmp_path / "seed"
    (seed / "certs").mkdir(parents=True)
    (seed / "acme").mkdir()
    (seed / "acme" / "account.key").write_text("existing-account")
    revision = "1234567890abcdef"
    (seed / "revision").write_text(revision + "\n")
    (seed / "primary").write_text("example.test\n")
    (seed / "domains").write_text("example.test\n*.example.test\n")
    key = seed / "certs/example.test_key.pem"
    chain = seed / "certs/example.test_fullchain.pem"
    subprocess.run([
        "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
        "-keyout", str(key), "-out", str(chain), "-days", "2",
        "-subj", "/CN=example.test",
        "-addext", "subjectAltName=DNS:example.test,DNS:*.example.test",
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    shutil.copyfile(str(chain), str(seed / "certs/example.test_cert.pem"))

    (tmp_path / "certs").mkdir()
    mock = tmp_path / "acme.sh"
    mock.write_text("""#!/bin/sh
set -eu
case " $* " in
  *" --issue "*) echo 'Unexpected runtime issuance' >&2; exit 41 ;;
esac
while test "$#" -gt 0; do
  case "$1" in
    --cert-file) cp "$MOCK_SEED/certs/example.test_cert.pem" "$2"; shift 2 ;;
    --key-file) cp "$MOCK_SEED/certs/example.test_key.pem" "$2"; shift 2 ;;
    --fullchain-file) cp "$MOCK_SEED/certs/example.test_fullchain.pem" "$2"; shift 2 ;;
    *) shift ;;
  esac
done
""")
    mock.chmod(0o755)

    script = container.get_source_path("nginx-certificates").read_text(encoding="utf-8")
    script = script.replace("/etc/certs", str(tmp_path / "certs"))
    script = script.replace("/opt/nginx-initial", str(seed))
    script = script.replace("/opt/acme/acme.sh", str(mock))
    script = script.replace("/usr/local/bin/nginx-acme", str(mock))
    path = tmp_path / "nginx-certificates"
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)
    env = dict(os.environ, MOCK_SEED=str(seed), NGINX_HTTPS_PORT="443")
    return container, path, tmp_path / "certs", seed, env, revision


def run_script(case, action):
    _, path, _, _, env, _ = case
    return subprocess.run([str(path), action], env=env, capture_output=True, text=True)


def test_acme_issuance_happens_in_image_build_only(certificate_case):
    container, _, _, _, _, _ = certificate_case
    dockerfile = container.docker_file
    assert "FROM alpine:3.23.6 AS acme" in dockerfile
    assert "FROM nginx:stable-alpine AS nginx" in dockerfile
    assert "COPY --from=acme /opt/acme /opt/acme" in dockerfile
    assert "COPY --from=acme /opt/nginx-initial /opt/nginx-initial" in dockerfile
    final_stage = dockerfile.split(" AS nginx", 1)[1]
    assert final_stage.index("--upgrade") < final_stage.index("--install-cert") < final_stage.index("--renew")
    assert "--renew --domain example.test" in final_stage
    assert "--upgrade --force" in final_stage
    assert "--force" not in final_stage.split("--renew", 1)[1]
    assert "--issue" not in final_stage
    assert "id=cntr_acme_account" not in final_stage
    assert "id=cntr_acme_dns" in final_stage
    assert "--issue --force" in dockerfile
    assert "--mount=type=secret,id=cntr_acme_dns" in dockerfile
    assert "ARG CNTR_BUILD_REVISION" in dockerfile
    assert "ENV CF_Token" not in dockerfile
    assert "nginx-certificates renew" in dockerfile
    spec = container.docker_compose["services"]["nginx"]
    assert spec["build"]["secrets"] == ["cntr_acme_account", "cntr_acme_dns"]


def test_offline_install_preserves_baked_cert_and_account(certificate_case):
    _, _, base, seed, _, revision = certificate_case
    result = run_script(certificate_case, "install")
    assert result.returncode == 0, result.stderr
    current = base / revision / "live"
    assert current.is_symlink()
    assert (current / "example.test_fullchain.pem").read_bytes() == (
        seed / "certs/example.test_fullchain.pem").read_bytes()
    assert (current / "acme/account.key").read_text() == "existing-account"
    assert (current / "renewal/example.test_key.pem").is_file()


def test_repeated_install_keeps_same_certificate_generation(certificate_case):
    _, _, base, _, _, revision = certificate_case
    assert run_script(certificate_case, "install").returncode == 0
    current = base / revision / "live"
    original = os.readlink(str(current))
    assert run_script(certificate_case, "install").returncode == 0
    assert os.readlink(str(current)) == original


def test_invalid_baked_key_never_publishes_live_certificate(certificate_case):
    _, _, base, seed, _, revision = certificate_case
    (seed / "certs/example.test_key.pem").unlink()
    result = run_script(certificate_case, "install")
    assert result.returncode != 0
    assert not (base / revision / "live").exists()


def test_invalid_san_never_publishes_certificate(certificate_case):
    _, _, base, seed, _, revision = certificate_case
    (seed / "domains").write_text("example.test\n*.absent.test\n")
    result = run_script(certificate_case, "install")
    assert result.returncode != 0
    assert not (base / revision / "live").exists()


def test_check_uses_revision_specific_state(certificate_case):
    _, _, base, seed, _, revision = certificate_case
    assert run_script(certificate_case, "check").returncode == 0
    assert (base / revision / "live").is_symlink()
    assert (base / revision / "live/example.test_key.pem").is_file()
    assert (seed / "acme/account.key").read_text() == "existing-account"


def test_https_validation_uses_writable_disposable_certificate_tree(certificate_case, monkeypatch):
    container, _, base, _, _, _ = certificate_case
    assert any(mount.endswith(":/etc/certs")
               for mount in container.docker_compose["services"]["nginx"]["volumes"])
    generated = base.parent / "generated"
    credentials = base.parent / "dns.env"
    model = {"services": {"nginx": {"image": "nginx:target", "volumes": [
        {"type": "bind", "source": str(base), "target": "/etc/certs"},
        {"type": "bind", "source": str(generated), "target": "/etc/nginx/managed", "read_only": True},
        {"type": "bind", "source": str(credentials), "target": "/run/acme-secrets", "read_only": True},
    ]}}}
    observed = []

    def execute(args, **kwargs):
        mounts = [args[index + 1] for index, arg in enumerate(args) if arg == "--mount"]
        certs = next(mount for mount in mounts if "target=/etc/certs" in mount)
        assert not certs.endswith(",readonly")
        source = Path(certs.split("source=", 1)[1].split(",", 1)[0])
        assert source != base
        assert not any("source=" + str(base) + "," in mount for mount in mounts)
        assert (source / container.cert_image_revision / "previous").is_dir()
        assert "type=bind,source={},target=/etc/nginx/managed,readonly".format(generated) in mounts
        assert "type=bind,source={},target=/run/acme-secrets,readonly".format(credentials) in mounts
        assert "/usr/local/bin/nginx-certificates check && " in args[-1]
        observed.append(source)
        return SimpleNamespace(succeeded=True, stdout="", stderr="", returncode=0)

    monkeypatch.setattr(container.runtime, "create_docker_process", lambda *args, **kwargs: args)
    monkeypatch.setattr(container.manager.structured_runner, "execute", execute)
    container.on_check(SimpleNamespace(compose_model=model))
    assert len(observed) == 1
    assert not observed[0].exists()


def test_new_revision_does_not_modify_older_certificate(certificate_case):
    _, _, base, seed, _, revision = certificate_case
    assert run_script(certificate_case, "install").returncode == 0
    first = base / revision / "live"
    original_link = os.readlink(str(first))
    (seed / "revision").write_text("fedcba0987654321\n")
    assert run_script(certificate_case, "install").returncode == 0
    assert os.readlink(str(first)) == original_link
    assert (base / "fedcba0987654321/live").is_symlink()


@pytest.mark.parametrize("source", ["target", "running", "seed", "empty"])
def test_build_account_uses_current_storage_without_migrating_old_files(fresh_manager, monkeypatch, source):
    container = fresh_manager.containers["nginx"]
    fresh_manager.env_config.set("NGINX_HTTPS_ENABLE", True)
    revision, running_revision = "1234567890abcdef", "fedcba0987654321"
    container.__dict__.update(cert_image_revision=revision, extend_configs={}, _rendered_site_files=({}, False))
    accounts = {
        "target": container.get_app_path("certs", revision, "live", "acme"),
        "running": container.get_app_path("certs", running_revision, "live", "acme"),
        "seed": container.get_app_path("acme"),
    }
    available = {"target": ("target", "running", "seed"), "running": ("running", "seed"),
                 "seed": ("seed",), "empty": ()}
    for name in available[source]:
        if name in accounts:
            accounts[name].mkdir(parents=True, exist_ok=True)
            (accounts[name] / "account.conf").write_text(name)
    old_files = [container.get_app_path(name, "untouched")
                 for name in ("conf.d", "migration-backup-existing", "certs/live/acme")]
    for path in old_files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("old layout")

    def fail_process(*args, **kwargs):
        raise AssertionError("Preparation must not copy from or change a container")

    monkeypatch.setattr(container.runtime, "create_docker_process", fail_process)
    monkeypatch.setattr(container.runtime, "chmod", lambda *args: None)
    monkeypatch.setattr(container, "_render_site_template", lambda *args: "configuration")
    context = SimpleNamespace(
        initial_runtime_state=SimpleNamespace(services=(SimpleNamespace(service="nginx", labels={
            "io.linktools.nginx.certificate-revision": running_revision,
        }),)),
        previous_compose_contents={"nginx": "original model"}, write_files=lambda *args: None,
    )
    container.on_starting(context)
    with tarfile.open(str(container.get_app_path("acme-build-account.tar"))) as archive:
        if source == "empty":
            assert archive.getnames() == []
        else:
            assert archive.extractfile("account.conf").read().decode() == source
    assert context.previous_compose_contents == {"nginx": "original model"}
    assert all(path.read_text() == "old layout" for path in old_files)
    assert list(container.get_app_path().glob("migration-backup-*")) == [old_files[1].parent]


def test_install_uses_image_seed_without_importing_flat_layout(certificate_case):
    _, _, base, seed, _, revision = certificate_case
    old = base / "old"
    shutil.copytree(str(seed / "certs"), str(old))
    shutil.copytree(str(seed / "acme"), str(old / "acme"))
    (old / "acme/account.key").write_text("old-layout-account")
    (old / "build-revision").write_text(revision)
    (base / "live").symlink_to("old", target_is_directory=True)
    result = run_script(certificate_case, "install")
    assert result.returncode == 0, result.stderr
    assert (base / revision / "live/acme/account.key").read_text() == "existing-account"
    assert (old / "acme/account.key").read_text() == "old-layout-account"
    assert os.readlink(str(base / "live")) == "old"


@pytest.mark.parametrize("failure", ("download", "install", "missing-executable", None))
def test_acme_install_stage_owns_home_and_propagates_failure(certificate_case, tmp_path, failure):
    container = certificate_case[0]
    dockerfile = container.docker_file
    command = dockerfile.split("RUN curl ", 1)[1].split("\nCOPY ", 1)[0]
    command = "curl " + command.replace("\\\n", "")
    work = tmp_path / "work"
    work.mkdir()
    home = tmp_path / "installed"
    config = tmp_path / "config"
    command = command.replace("/tmp/", str(work) + "/").replace("-C /tmp ", "-C " + str(work) + " ")
    command = command.replace("/opt/acme", str(home)).replace("/root/.acme.sh", str(config))
    source = tmp_path / "source" / "acme.sh-master"
    source.mkdir(parents=True)
    (source / "acme.sh").write_text('''#!/bin/sh
set -eu
test "$1" = --install
shift
while test "$#" -gt 0; do
  case "$1" in
    --home) home=$2; shift 2 ;;
    --config-home) config=$2; shift 2 ;;
    --nocron|--noprofile) shift ;;
    *) exit 91 ;;
  esac
done
test "$MOCK_FAILURE" != install || exit 92
test "$MOCK_FAILURE" != missing-executable || exit 0
mkdir -p "$home" "$config"
printf '#!/bin/sh\\nexit 0\\n' > "$home/acme.sh"
chmod +x "$home/acme.sh"
''')
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as package:
        package.add(source, arcname="acme.sh-master")
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "curl").write_text('''#!/bin/sh
set -eu
test "$MOCK_FAILURE" != download || exit 93
test "$1" = -fsSL
while test "$#" -gt 0; do
  if test "$1" = -o; then cp "$MOCK_ARCHIVE" "$2"; exit 0; fi
  shift
done
exit 94
''')
    (tools / "curl").chmod(0o755)
    env = dict(os.environ, PATH=str(tools) + os.pathsep + os.environ["PATH"],
               MOCK_ARCHIVE=str(archive), MOCK_FAILURE=failure or "")
    result = subprocess.run(["sh", "-c", command], env=env, capture_output=True, text=True)
    if failure:
        assert result.returncode != 0
        assert not (home / "dns-keys").exists()
        return
    assert result.returncode == 0, result.stderr
    assert os.access(home / "acme.sh", os.X_OK)
    assert config.is_dir()
    assert not (work / "acme.sh-master").exists()
    assert not (work / "acme.tar.gz").exists()
    keys = next(line[4:] for line in dockerfile.splitlines() if line.startswith("RUN printf '%s\\n'"))
    keys = keys.replace("/opt/acme", str(home))
    subprocess.run(["sh", "-c", keys], check=True, env=env)
    assert "CF_Token" in (home / "dns-keys").read_text().splitlines()
    local_bin = tmp_path / "usr-local-bin"
    usr_bin = tmp_path / "usr-bin"
    local_bin.mkdir()
    usr_bin.mkdir()
    for name in ("nginx-certificates", "nginx-acme"):
        shutil.copyfile(container.get_source_path(name), local_bin / name)
    setup = "chmod +x " + dockerfile.split("RUN chmod +x ", 1)[1].split("\n\nARG ", 1)[0]
    setup = setup.replace("\\\n", "").replace("/usr/local/bin", str(local_bin))
    setup = setup.replace("/usr/bin", str(usr_bin)).replace("/opt/acme", str(home))
    subprocess.run(["sh", "-c", setup], check=True, env=env)
    assert (usr_bin / "acme.sh").resolve() == home / "acme.sh"
    assert all(os.access(local_bin / name, os.X_OK) for name in ("nginx-certificates", "nginx-acme"))


@pytest.mark.parametrize("waf", [False, True])
def test_http_only_build_does_not_require_acme(fresh_manager, waf):
    fresh_manager.env_config.set("NGINX_HTTPS_ENABLE", False)
    fresh_manager.env_config.set("NGINX_WAF_ENABLE", waf)
    container = fresh_manager.containers["nginx"]
    container.reset_rendered_inputs()
    dockerfile = container.docker_file
    assert "FROM nginx:stable-alpine AS nginx" in dockerfile
    assert "FROM alpine:" not in dockerfile
    assert "nginx-acme" not in dockerfile
    assert "secrets" not in container.docker_compose["services"]["nginx"]["build"]


@pytest.mark.parametrize("renew_status", [0, 2, 1, 3])
@pytest.mark.parametrize("install_status", [0, 1])
@pytest.mark.parametrize("upgrade_status", [0, 1, 2])
def test_final_stage_refresh_accepts_only_success_or_not_due(certificate_case, tmp_path,
                                                            renew_status, install_status, upgrade_status):
    container, _, _, seed, _, _ = certificate_case
    final = container.docker_file.split(" AS nginx", 1)[1]
    command = "set -eu;" + final.rsplit("set -eu;", 1)[1]
    command = command.replace("\\\n", "").replace("/opt/nginx-initial", str(seed))
    calls = tmp_path / "refresh-calls"
    wrapper = tmp_path / "refresh-acme"
    wrapper.write_text("""#!/bin/sh
set -eu
printf '%s\\n' "$*" >> "$MOCK_CALLS"
case " $* " in
  *" --upgrade "*) exit "$MOCK_UPGRADE_STATUS" ;;
  *" --install-cert "*) exit "$MOCK_INSTALL_STATUS" ;;
  *" --renew "*) exit "$MOCK_RENEW_STATUS" ;;
  *) exit 91 ;;
esac
""")
    wrapper.chmod(0o755)
    command = command.replace("/usr/local/bin/nginx-acme", str(wrapper))
    env = dict(os.environ, CNTR_BUILD_REVISION="test-only", MOCK_CALLS=str(calls),
               MOCK_INSTALL_STATUS=str(install_status), MOCK_RENEW_STATUS=str(renew_status),
               MOCK_UPGRADE_STATUS=str(upgrade_status))
    result = subprocess.run(["sh", "-c", command], env=env, capture_output=True, text=True)
    assert (result.returncode == 0) is (upgrade_status == 0 and install_status == 0 and renew_status in (0, 2))
    seen = calls.read_text().splitlines()
    assert "--upgrade --force" in seen.pop(0)
    if upgrade_status:
        assert seen == []
        return
    assert "--install-cert --domain example.test" in seen[0]
    assert "--reloadcmd :" in seen[0]
    assert str(seed / "certs/example.test_fullchain.pem") in seen[0]
    assert len(seen) == (1 if install_status else 2)
    if not install_status:
        assert "--renew --domain example.test" in seen[1]
    assert not any("--force" in call or "--issue" in call for call in seen)


@pytest.mark.parametrize("renew_status", [0, 2, 1])
def test_final_refresh_does_not_persist_dns_credentials(certificate_case, tmp_path, renew_status):
    container, _, _, seed, _, _ = certificate_case
    credentials = tmp_path / "dns.env"
    credentials.write_text("export CF_Token='test-current-secret'\n")
    keys = tmp_path / "dns-keys"
    keys.write_text("CF_Token\nAli_Key\n")
    account = seed / "acme/account.conf"
    account.write_text("SAVED_Ali_Key='test-old-secret'\nACCOUNT_EMAIL='test@example.test'\n")
    mock = tmp_path / "mock-acme"
    mock.write_text("""#!/bin/sh
set -eu
test "$CF_Token" = test-current-secret
! grep -q test-old-secret "$MOCK_ACCOUNT"
printf "SAVED_CF_Token='test-current-secret'\\n" >> "$MOCK_ACCOUNT"
case " $* " in
  *" --renew "*) exit "$MOCK_RENEW_STATUS" ;;
  *" --upgrade "*|*" --install-cert "*) exit 0 ;;
  *) exit 91 ;;
esac
""")
    mock.chmod(0o755)
    wrapper = tmp_path / "actual-nginx-acme"
    wrapper.write_text(container.get_source_path("nginx-acme").read_text()
                       .replace("/run/acme-secrets/dns.env", str(credentials))
                       .replace("/opt/acme/dns-keys", str(keys))
                       .replace("/opt/acme/acme.sh", str(mock)))
    wrapper.chmod(0o755)
    command = "set -eu;" + container.docker_file.rsplit("set -eu;", 1)[1]
    command = (command.replace("\\\n", "").replace("/opt/nginx-initial", str(seed))
               .replace("/usr/local/bin/nginx-acme", str(wrapper)))
    env = dict(os.environ, CNTR_BUILD_REVISION="test-only", MOCK_ACCOUNT=str(account),
               MOCK_RENEW_STATUS=str(renew_status))
    result = subprocess.run(["sh", "-c", command], env=env, capture_output=True, text=True)
    assert (result.returncode == 0) is (renew_status in (0, 2)), result.stderr
    assert account.read_text() == "ACCOUNT_EMAIL='test@example.test'\n"
    assert (seed / "acme/account.key").read_text() == "existing-account"
