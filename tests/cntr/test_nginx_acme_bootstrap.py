#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build-time ACME inputs and offline certificate installation with the current script."""

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from linktools.cntr import OperationContext


@pytest.fixture
def certificate_case(fresh_manager, tmp_path):
    container = fresh_manager.containers["nginx"]
    settings = fresh_manager.config
    settings.set("NGINX_ROOT_DOMAIN", "example.test")
    settings.set("NGINX_HTTPS_ENABLE", True)
    settings.set("ACME_DNS_API", "dns_cf")
    settings.set("CF_Token", "fake-secret")
    container.__dict__["acme_ssl_domains"] = ["example.test", "*.example.test"]
    container.__dict__["sites"] = {}
    for name in ("docker_file", "docker_compose"):
        container.__dict__.pop(name, None)

    seed = tmp_path / "seed"
    (seed / "certs").mkdir(parents=True)
    (seed / "acme").mkdir()
    (seed / "acme" / "account.key").write_text("existing-account")
    settings_text = "letsencrypt\ndns_cf\n\n"
    (seed / "settings").write_text(settings_text)
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
    return container, path, tmp_path / "certs", seed, env, settings_text


def run_script(case, action):
    _, path, _, _, env, _ = case
    return subprocess.run([str(path), action], env=env, capture_output=True, text=True)


def test_acme_issuance_happens_in_image_build_only(certificate_case):
    container, _, _, _, _, _ = certificate_case
    dockerfile = container.docker_file
    assert "AS acme-build" not in dockerfile
    assert "--issue --force" in dockerfile
    assert "--mount=type=secret,id=cntr_acme_dns" in dockerfile
    assert "ENV CF_Token" not in dockerfile
    assert "nginx-certificates renew" in dockerfile
    spec = container.docker_compose["services"]["nginx"]
    assert spec["build"]["secrets"] == ["cntr_acme_account", "cntr_acme_dns"]


def test_build_step_consumes_native_arguments_without_shell_evaluation(certificate_case, tmp_path):
    container = certificate_case[0]
    command = container.docker_file.split("<<'BUILD_CERTIFICATES'\n", 1)[1].split("\nBUILD_CERTIFICATES", 1)[0]
    home, certs, output = (tmp_path / name for name in ("build-acme", "build-certs", "build-seed"))
    mock = tmp_path / "build-acme.sh"
    mock.write_text('''#!/bin/sh
printf '%s\\n' "$@" >> "$BUILD_LOG"
mkdir -p "$1"
while test "$#" -gt 0; do
  case "$1" in
    --cert-file|--key-file|--fullchain-file) printf 'fixture' > "$2"; shift 2 ;;
    *) shift ;;
  esac
done
''')
    mock.chmod(0o755)
    command = command.replace("/usr/local/bin/nginx-acme", str(mock))
    for source, target in (("/root/.acme.sh", home), ("/etc/certs", certs),
                           ("/opt/nginx-initial", output),
                           ("/run/secrets/cntr_acme_account", tmp_path / "absent-account")):
        command = command.replace(source, str(target))
    marker = tmp_path / "must-not-exist"
    server = "server; touch " + str(marker)
    email = "$(touch " + str(marker) + ")@example.test"
    env = dict(os.environ, NGINX_ROOT_DOMAIN="example.test",
               NGINX_CERT_DOMAINS="example.test\n*.example.test", ACME_SERVER=server,
               ACME_DNS_API="dns_cf", ACME_ACCOUNT_EMAIL=email, ACME_DNS_CREDENTIALS_DIGEST="fixture-digest", BUILD_LOG=str(tmp_path / "build.log"))
    result = subprocess.run(["sh", "-c", command], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert not marker.exists()
    assert (output / "primary").read_text() == "example.test\n"
    assert (output / "domains").read_text() == "example.test\n*.example.test\n"
    assert (output / "settings").read_text() == server + "\ndns_cf\n" + email + "\n"
    arguments = (tmp_path / "build.log").read_text().splitlines()
    assert "*.example.test" in arguments
    assert server in arguments and email in arguments
    assert not (output / "revision").exists()


@pytest.mark.parametrize("https", [False, True])
def test_builtin_nginx_build_uses_native_arguments(fresh_manager, https):
    from linktools.cntr.runtime.images import ImagePreparer, LocalImage
    container = fresh_manager.containers["nginx"]
    fresh_manager.config.set("NGINX_ROOT_DOMAIN", "example.test")
    fresh_manager.config.set("NGINX_HTTPS_ENABLE", https)
    fresh_manager.config.set("ACME_DNS_API", "dns_cf")
    container.__dict__["sites"] = {}
    dockerfile = container.docker_file
    raw = container.docker_compose
    assert raw["services"]["nginx"]["image"] == container.get_service_name("nginx")
    assert "labels" not in raw["services"]["nginx"]["build"]
    prep = fresh_manager.image_preparer
    model = prep.with_build_revisions(raw, [container], ["nginx"])
    spec = model["services"]["nginx"]
    revision = spec["build"]["labels"][ImagePreparer.BUILD_LABEL]
    assert len(revision) == 64
    assert spec["build"]["args"]["NGINX_TAG"] == "stable-alpine"
    assert "LABEL " not in dockerfile
    assert "revision" not in dockerfile
    if https:
        assert spec["build"]["args"]["NGINX_CERT_DOMAINS"] == "example.test\n*.example.test"
        assert "--issue --force" in dockerfile
        assert '"$NGINX_CERT_DOMAINS"' in dockerfile
        assert "CF_Token" not in spec["build"]["args"]
        container.__dict__["acme_ssl_domains"] = ["example.test", "*.example.test", "other.test"]
        container.__dict__.pop("docker_compose", None)
        changed = prep.with_build_revisions(container.docker_compose, [container], ["nginx"])
        assert container.docker_file == dockerfile
        assert changed["services"]["nginx"]["build"]["labels"][ImagePreparer.BUILD_LABEL] != revision
        prep.inspect_images = lambda images: {image: LocalImage("sha256:old", revision) for image in images}
        assert prep.plan(changed).build == ("nginx",)
    else:
        assert "--issue" not in dockerfile
        assert "secrets" not in spec["build"]
        assert spec["build"]["args"] == {"NGINX_TAG": "stable-alpine"}



@pytest.mark.parametrize("https", [False, True])
def test_nginx_native_defaults_match_explicit_build_fields(fresh_manager, monkeypatch, https):
    import yaml
    container = fresh_manager.containers["nginx"]
    fresh_manager.config.set("NGINX_ROOT_DOMAIN", "example.test")
    fresh_manager.config.set("NGINX_HTTPS_ENABLE", https)
    fresh_manager.config.set("ACME_DNS_API", "dns_cf")
    container.__dict__["sites"] = {}
    implicit = container.docker_compose
    original_render = container.render_template

    def explicit_render(source, *args, **kwargs):
        text = original_render(source, *args, **kwargs)
        if Path(source).name == "compose.yml":
            model = yaml.safe_load(text)
            spec = model["services"]["nginx"]
            spec["image"] = container.get_service_name("nginx")
            spec["build"]["context"] = str(container.get_docker_context_path())
            spec["build"]["dockerfile"] = str(container.get_docker_file_destination())
            return yaml.safe_dump(model)
        return text

    monkeypatch.setattr(container, "render_template", explicit_render)
    container.__dict__.pop("docker_compose", None)
    explicit = container.docker_compose
    assert implicit == explicit
    prep = fresh_manager.image_preparer
    assert prep.with_build_revisions(implicit, [container], ["nginx"]) == (
        prep.with_build_revisions(explicit, [container], ["nginx"]))
    assert not any("/root/.acme.sh" in mount for mount in implicit["services"]["nginx"]["volumes"])


def test_dns_credential_digest_tracks_only_dns_payload(fresh_manager, monkeypatch):
    from linktools.cntr.runtime.images import ImagePreparer, LocalImage
    container = fresh_manager.containers["nginx"]
    config = fresh_manager.config
    config.set("NGINX_ROOT_DOMAIN", "example.test")
    config.set("NGINX_HTTPS_ENABLE", True)
    config.set("ACME_DNS_API", "dns_cf")
    config.set("CF_Token", "test-only-token-one 'quoted'")
    container.__dict__.update(sites={}, _rendered_site_files=({}, False))
    monkeypatch.setattr(container.runtime, "chmod", lambda *args: None)
    prep = fresh_manager.image_preparer

    def render():
        container.__dict__.pop("docker_compose", None)
        return prep.with_build_revisions(container.docker_compose, [container], ["nginx"])

    original = render()
    spec = original["services"]["nginx"]
    digest = spec["build"]["args"]["ACME_DNS_CREDENTIALS_DIGEST"]
    revision = spec["build"]["labels"][ImagePreparer.BUILD_LABEL]
    assert not container.get_app_path("acme-secrets", "dns.env").exists()
    assert digest == hashlib.sha256(container.acme_dns_environment.encode("utf-8")).hexdigest()
    assert len(digest) == 64
    assert "test-only-token-one" not in json.dumps(original) + container.docker_file
    assert 'test -n "$ACME_DNS_CREDENTIALS_DIGEST"' in container.docker_file.split("<<'BUILD_CERTIFICATES'", 1)[1]
    container.on_starting(OperationContext())
    payload = container.get_app_path("acme-secrets", "dns.env").read_bytes()
    assert hashlib.sha256(payload).hexdigest() == digest
    assert b"test-only-token-one" in payload
    account = container.get_app_path("acme", "account.key")
    account.write_text("test-only-account-key")
    container.on_starting(OperationContext())
    assert render() == original
    prep.inspect_images = lambda images: {image: LocalImage("sha256:old", revision) for image in images}
    assert prep.plan(original).build == ()
    config.set("CF_Token", "test-only-token-two")
    # A resolved command keeps the secret paired with its rendered digest.
    container.on_starting(OperationContext())
    assert container.get_app_path("acme-secrets", "dns.env").read_bytes() == payload
    assert render() == original
    # A new container instance represents the next command's input snapshot.
    container = type(container)(fresh_manager, container.root_path, name="nginx")
    container.__dict__.update(sites={}, _rendered_site_files=({}, False))
    changed = render()
    changed_spec = changed["services"]["nginx"]
    assert changed_spec["image"] == spec["image"]
    assert changed_spec["build"]["args"]["ACME_DNS_CREDENTIALS_DIGEST"] != digest
    assert changed_spec["build"]["labels"][ImagePreparer.BUILD_LABEL] != revision
    assert "test-only-token-two" not in json.dumps(changed) + container.docker_file
    assert prep.plan(changed).build == ("nginx",)
    container.on_starting(OperationContext())
    assert hashlib.sha256(container.get_app_path("acme-secrets", "dns.env").read_bytes()).hexdigest() == (
        changed_spec["build"]["args"]["ACME_DNS_CREDENTIALS_DIGEST"])


def test_http_build_does_not_resolve_dns_credentials(fresh_manager, monkeypatch):
    container = fresh_manager.containers["nginx"]
    fresh_manager.config.set("NGINX_HTTPS_ENABLE", False)
    container.__dict__["sites"] = {}

    def unexpected(_):
        raise AssertionError("HTTP-only build resolved DNS credentials")

    monkeypatch.setattr(type(container), "acme_dns_environment", property(unexpected))
    spec = container.docker_compose["services"]["nginx"]
    assert "ACME_DNS_CREDENTIALS_DIGEST" not in spec["build"]["args"]
    assert "ACME_DNS_CREDENTIALS_DIGEST" not in container.docker_file


@pytest.mark.parametrize("script", ["nginx-certificates", "nginx-acme"])
def test_builtin_script_changes_require_explicit_build(fresh_manager, tmp_path, monkeypatch, script):
    from linktools.cntr.runtime.images import ImagePreparer, LocalImage
    container = fresh_manager.containers["nginx"]
    fresh_manager.config.set("NGINX_ROOT_DOMAIN", "example.test")
    fresh_manager.config.set("NGINX_HTTPS_ENABLE", True)
    fresh_manager.config.set("ACME_DNS_API", "dns_cf")
    container.__dict__["sites"] = {}
    source = tmp_path / "nginx"
    shutil.copytree(str(container.get_source_path()), str(source))
    monkeypatch.setattr(container, "get_source_path", lambda *parts: source.joinpath(*parts))
    prep = fresh_manager.image_preparer
    original = prep.with_build_revisions(container.docker_compose, [container], ["nginx"])
    spec = original["services"]["nginx"]
    revision = spec["build"]["labels"][ImagePreparer.BUILD_LABEL]
    content = (source / script).read_text()
    assert "COPY nginx-certificates nginx-acme /usr/local/bin/" in container.docker_file
    assert "chmod +x /usr/local/bin/nginx-certificates /usr/local/bin/nginx-acme" in container.docker_file
    prep.inspect_images = lambda images: {image: LocalImage("sha256:old", revision) for image in images}
    assert prep.plan(original).build == ()
    (source / script).write_text(content + "# Script implementation changed\n")
    container.__dict__.pop("docker_file", None)
    changed = prep.with_build_revisions(container.docker_compose, [container], ["nginx"])
    assert changed["services"]["nginx"]["image"] == spec["image"]
    assert changed["services"]["nginx"]["build"]["args"] == spec["build"]["args"]
    assert changed["services"]["nginx"]["build"]["labels"][ImagePreparer.BUILD_LABEL] == revision
    assert prep.plan(changed).build == ()
    assert prep.plan(changed, force_pull=True).build == ("nginx",)


def test_offline_install_preserves_baked_cert_and_account(certificate_case):
    _, _, base, seed, _, _ = certificate_case
    result = run_script(certificate_case, "install")
    assert result.returncode == 0, result.stderr
    current = base / "live"
    assert current.is_symlink()
    assert (current / "example.test_fullchain.pem").read_bytes() == (
        seed / "certs/example.test_fullchain.pem").read_bytes()
    assert (current / "acme/account.key").read_text() == "existing-account"
    assert (current / "renewal/example.test_key.pem").is_file()


def test_repeated_install_keeps_same_certificate_generation(certificate_case):
    _, _, base, _, _, _ = certificate_case
    assert run_script(certificate_case, "install").returncode == 0
    current = base / "live"
    original = os.readlink(str(current))
    assert run_script(certificate_case, "install").returncode == 0
    assert os.readlink(str(current)) == original


def test_invalid_baked_key_never_publishes_live_certificate(certificate_case):
    _, _, base, seed, _, _ = certificate_case
    (seed / "certs/example.test_key.pem").unlink()
    result = run_script(certificate_case, "install")
    assert result.returncode != 0
    assert not (base / "live").exists()


def test_invalid_san_never_publishes_certificate(certificate_case):
    _, _, base, seed, _, _ = certificate_case
    (seed / "domains").write_text("example.test\n*.absent.test\n")
    result = run_script(certificate_case, "install")
    assert result.returncode != 0
    assert not (base / "live").exists()


def test_check_installs_valid_seed_without_account_mutation(certificate_case):
    _, _, base, seed, _, _ = certificate_case
    assert run_script(certificate_case, "check").returncode == 0
    assert (base / "live").is_symlink()
    assert (base / "live/example.test_key.pem").is_file()
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
        assert (source / "previous").is_dir()
        assert "type=bind,source={},target=/etc/nginx/managed,readonly".format(generated) in mounts
        assert "type=bind,source={},target=/run/acme-secrets,readonly".format(credentials) in mounts
        assert "/usr/local/bin/nginx-certificates check && " in args[-1]
        observed.append(source)
        return SimpleNamespace(succeeded=True, stdout="", stderr="", returncode=0)

    monkeypatch.setattr(container.runtime, "create_docker_process", lambda *args, **kwargs: args)
    monkeypatch.setattr(container.manager.structured_runner, "execute", execute)
    container.__dict__["config_path"] = generated
    container.on_check(OperationContext(compose_model=model))
    assert len(observed) == 1
    assert not observed[0].exists()


@pytest.mark.parametrize("changed_settings", ["other-ca\ndns_cf\n\n", "letsencrypt\ndns_other\n\n", "letsencrypt\ndns_cf\nnew@example.test\n"])
def test_parameter_change_and_image_rollback_preserve_existing_generation(certificate_case, changed_settings):
    _, _, base, seed, _, original_settings = certificate_case
    assert run_script(certificate_case, "install").returncode == 0
    original_link = os.readlink(str(base / "live"))
    original = base / original_link
    (original / "acme/account.key").write_text("renewed-account")
    # The old image's seed can differ from its subsequently renewed account state.
    (seed / "settings").write_text(changed_settings)
    assert run_script(certificate_case, "install").returncode == 0
    replacement = os.readlink(str(base / "live"))
    assert replacement != original_link
    assert (original / "acme/account.key").read_text() == "renewed-account"
    (seed / "settings").write_text(original_settings)
    assert run_script(certificate_case, "install").returncode == 0
    assert os.readlink(str(base / "live")) == original_link
    assert (base / "live/acme/account.key").read_text() == "renewed-account"
    assert (base / replacement).is_dir()


@pytest.mark.parametrize("fail_reload", [False, True])
def test_renewal_verifies_served_certificate_and_rolls_back(certificate_case, tmp_path, fail_reload):
    _, _, base, seed, env, _ = certificate_case
    assert run_script(certificate_case, "install").returncode == 0
    previous = os.readlink(str(base / "live"))
    previous_chain = (base / previous / "example.test_fullchain.pem").read_bytes()
    renewal = base / previous / "renewal"
    subprocess.run([
        "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
        "-keyout", str(renewal / "example.test_key.pem"),
        "-out", str(renewal / "example.test_fullchain.pem"), "-days", "3",
        "-subj", "/CN=example.test",
        "-addext", "subjectAltName=DNS:example.test,DNS:*.example.test",
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    renewed_chain = (renewal / "example.test_fullchain.pem").read_bytes()
    tools = tmp_path / "tools"
    tools.mkdir()
    real_openssl = shutil.which("openssl")
    (tools / "openssl").write_text(
        '#!/bin/sh\nif test "$1" = s_client; then cat "$SERVED_CHAIN"; else exec "' +
        real_openssl + '" "$@"; fi\n')
    (tools / "nginx").write_text('''#!/bin/sh
case " $* " in
  *" -s reload "*)
    if test "$FAIL_RELOAD" = 1 && ! test -f "$RELOAD_ATTEMPT"; then
      touch "$RELOAD_ATTEMPT"
      exit 1
    fi ;;
esac
''')
    for name in ("openssl", "nginx"):
        (tools / name).chmod(0o755)
    env.update(PATH=str(tools) + os.pathsep + env["PATH"], FAIL_RELOAD="1" if fail_reload else "0",
               RELOAD_ATTEMPT=str(tmp_path / "reload-attempt"),
               SERVED_CHAIN=str(base / previous / "example.test_fullchain.pem") if fail_reload else
               str(renewal / "example.test_fullchain.pem"))
    result = run_script(certificate_case, "renew")
    assert result.returncode == (1 if fail_reload else 0), result.stderr
    assert (os.readlink(str(base / "live")) == previous) is fail_reload
    assert (base / "live/example.test_fullchain.pem").read_bytes() == (previous_chain if fail_reload else renewed_chain)
    assert (base / previous / "example.test_fullchain.pem").read_bytes() == previous_chain
    assert (seed / "certs/example.test_fullchain.pem").read_bytes() == previous_chain
    if not fail_reload:
        renewed = os.readlink(str(base / "live"))
        original_settings = (seed / "settings").read_text()
        (seed / "settings").write_text("other-ca\ndns_cf\n\n")
        assert run_script(certificate_case, "install").returncode == 0
        (seed / "settings").write_text(original_settings)
        assert run_script(certificate_case, "install").returncode == 0
        assert os.readlink(str(base / "live")) == renewed
        assert (base / "live/example.test_fullchain.pem").read_bytes() == renewed_chain


@pytest.mark.parametrize("source", ["live", "seed", "empty"])
def test_build_account_uses_live_storage_without_revision_lookup(fresh_manager, monkeypatch, source):
    container = fresh_manager.containers["nginx"]
    fresh_manager.config.set("NGINX_HTTPS_ENABLE", True)
    container.__dict__.update(extra_fields={}, _rendered_site_files=({}, False))
    for name, path in (("live", container.get_app_path("certs", "live", "acme")),
                       ("seed", container.get_app_path("acme"))):
        if source == name or (source == "live" and name == "seed"):
            path.mkdir(parents=True, exist_ok=True)
            (path / "account.conf").write_text(name)
    old = container.get_app_path("certs", "1234567890abcdef", "live", "acme")
    old.mkdir(parents=True)
    (old / "account.conf").write_text("untouched")
    monkeypatch.setattr(container.runtime, "chmod", lambda *args: None)
    container.on_starting(OperationContext())
    with tarfile.open(str(container.get_app_path("acme-build-account.tar"))) as archive:
        if source == "empty":
            assert archive.getnames() == []
        else:
            assert archive.extractfile("account.conf").read().decode() == source
    assert (old / "account.conf").read_text() == "untouched"


def test_install_uses_image_seed_without_importing_flat_layout(certificate_case):
    _, _, base, seed, _, _ = certificate_case
    old = base / "old"
    shutil.copytree(str(seed / "certs"), str(old))
    shutil.copytree(str(seed / "acme"), str(old / "acme"))
    (old / "acme/account.key").write_text("old-layout-account")
    (old / "build-revision").write_text("old-unused-revision")
    (base / "live").symlink_to("old", target_is_directory=True)
    result = run_script(certificate_case, "install")
    assert result.returncode == 0, result.stderr
    assert (base / "live/acme/account.key").read_text() == "existing-account"
    assert (old / "acme/account.key").read_text() == "old-layout-account"
    assert os.readlink(str(base / "live")) != "old"


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
    setup = "chmod +x " + dockerfile.split("RUN chmod +x ", 1)[1].split("\n\n", 1)[0]
    setup = setup.replace("\\\n", "").replace("/usr/local/bin", str(local_bin))
    setup = setup.replace("/usr/bin", str(usr_bin)).replace("/opt/acme", str(home))
    subprocess.run(["sh", "-c", setup], check=True, env=env)
    assert (usr_bin / "acme.sh").resolve() == home / "acme.sh"
    assert all(os.access(local_bin / name, os.X_OK) for name in ("nginx-certificates", "nginx-acme"))
