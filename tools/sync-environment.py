#!/usr/bin/env python3
"""Synchronize the pinned CPU workspace or run it in the locked Docker image.

Bootstrap requires Python >= 3.10 and Git on Linux x86_64. No pip packages,
system package installation, Git staging, or host Python replacement is used.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "env/environment.lock.json"
CACHE = Path("artifacts/environment")
GIT = "/usr/bin/git"
SHA256 = re.compile(r"[0-9a-f]{64}")


class EnvironmentError(RuntimeError):
    pass


def sha256(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def owned_path(relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise EnvironmentError(f"unsafe repository path: {relative}")
    path = ROOT
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise EnvironmentError(f"repository path traverses a symlink: {path}")
    return path


def run(arguments, *, capture=False, env=None, timeout=None):
    if not capture:
        print("+ " + shlex.join(map(str, arguments)), flush=True)
    result = subprocess.run(
        list(map(str, arguments)), cwd=ROOT, check=True, text=True,
        stdout=subprocess.PIPE if capture else None,
        env=env, timeout=timeout,
    )
    return result.stdout.strip() if capture else ""


def git(*arguments, capture=True):
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("GIT_")}
    environment.update({
        "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "0",
    })
    # A caller's CA is a transport input, never a replacement for source hashes.
    if "GIT_SSL_CAINFO" in os.environ:
        environment["GIT_SSL_CAINFO"] = os.environ["GIT_SSL_CAINFO"]
    return run([GIT, "-c", "core.hooksPath=/dev/null", "-c",
                "core.fsmonitor=false", "-c", "http.version=HTTP/1.1",
                *arguments], capture=capture, env=environment)


def load_lock():
    value = json.loads(LOCK_PATH.read_text())
    if value["schema_version"] != "1.0" or value["platform"] != "linux/amd64":
        raise EnvironmentError("unsupported environment lock")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise EnvironmentError("this baseline supports Linux x86_64 only")
    for name in ("python", "git", "uv", "bazel"):
        if not SHA256.fullmatch(value[name]["sha256"]):
            raise EnvironmentError(f"invalid {name} SHA-256")
    docker = value["docker"]
    if not re.fullmatch(r"ubuntu@sha256:[0-9a-f]{64}", docker["base_image"]):
        raise EnvironmentError("the Docker base must use an immutable Ubuntu digest")
    if not re.fullmatch(r"[0-9]{8}T[0-9]{6}Z", docker["apt_snapshot"]):
        raise EnvironmentError("invalid Ubuntu snapshot timestamp")
    for package in docker["apt_packages"]:
        if not re.fullmatch(r"[a-z0-9][a-z0-9+.-]*(=[A-Za-z0-9.+:~_-]+)?", package):
            raise EnvironmentError(f"invalid APT package pin: {package}")
    if not docker["package_mirrors"] or any(
        url not in ("https://archive.ubuntu.com/ubuntu", "https://security.ubuntu.com/ubuntu")
        for url in docker["package_mirrors"]
    ):
        raise EnvironmentError("package mirrors must be official Ubuntu archives")
    if len(set(value["source_paths"])) != len(value["source_paths"]):
        raise EnvironmentError("duplicate source path")
    for path in value["source_paths"]:
        owned_path(path)
    owned_path(value["bazel"]["path"])
    baseline = json.loads((ROOT / "manifests/baseline.json").read_text())
    expected_sources = {item["path"] for item in baseline["repository"]["sources"].values()}
    if set(value["source_paths"]) != expected_sources:
        raise EnvironmentError("environment sources differ from the baseline source set")
    baseline_uv = baseline["toolchain"]["uv"]
    if (value["uv"]["sha256"] != baseline_uv["artifact_sha256"]
            or value["uv"]["size_bytes"] != baseline_uv["artifact_size_bytes"]
            or value["uv"]["version"] != baseline_uv["repository_pin"]):
        raise EnvironmentError("uv identity differs from the runtime baseline")
    if value["python"]["version"] != (ROOT / ".python-version").read_text().strip():
        raise EnvironmentError("Python version differs from .python-version")
    build_schema = json.loads((ROOT / "manifests/schema/build-jaxlib.schema.json").read_text())
    properties = build_schema["$defs"]["pythonTool"]["properties"]
    for field, recorded in (("path", "base_executable"), ("size_bytes", "base_size_bytes"),
                            ("sha256", "base_sha256"), ("version", "version")):
        if value["python"][field] != properties[recorded]["const"]:
            raise EnvironmentError("Python identity differs from the strict build contract")
    return value


def matches(path, identity):
    path = Path(path)
    return (path.is_file()
            and ("size_bytes" not in identity or path.stat().st_size == identity["size_bytes"])
            and sha256(path) == identity["sha256"])


def write_verified(target, data, identity, *, executable=False):
    if hashlib.sha256(data).hexdigest() != identity["sha256"]:
        raise EnvironmentError(f"download SHA-256 mismatch: {target.name}")
    if "size_bytes" in identity and len(data) != identity["size_bytes"]:
        raise EnvironmentError(f"download size mismatch: {target.name}")
    owned_path(target.relative_to(ROOT))
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temporary:
        temporary.write(data)
        temporary_path = Path(temporary.name)
    try:
        temporary_path.chmod(0o755 if executable else 0o644)
        os.replace(temporary_path, target)
    finally:
        temporary_path.unlink(missing_ok=True)


def download(url):
    if not url.startswith("https://"):
        raise EnvironmentError("downloads require HTTPS")
    print(f"download: {url}", flush=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        if not response.geturl().startswith("https://"):
            raise EnvironmentError("download redirected away from HTTPS")
        data = response.read(256 * 1024 * 1024 + 1)
    if len(data) > 256 * 1024 * 1024:
        raise EnvironmentError("tool download exceeds 256 MiB")
    return data


def ensure_uv(lock):
    target = owned_path(CACHE / "bin/uv")
    if target.exists():
        if not matches(target, lock["uv"]):
            raise EnvironmentError(f"existing tool has unexpected bytes: {target}")
        return target
    candidate = shutil.which("uv")
    if candidate and matches(candidate, lock["uv"]):
        data = Path(candidate).read_bytes()
    else:
        archive = download(lock["uv"]["url"])
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as source:
            member = source.getmember(lock["uv"]["member"])
            if not member.isfile() or member.size != lock["uv"]["size_bytes"]:
                raise EnvironmentError("invalid uv archive member")
            data = source.extractfile(member).read()
    write_verified(target, data, lock["uv"], executable=True)
    return target


def ensure_bazel(lock):
    target = owned_path(lock["bazel"]["path"])
    if target.exists():
        if not matches(target, lock["bazel"]) or not os.access(target, os.X_OK):
            raise EnvironmentError(f"existing Bazel has unexpected bytes or mode: {target}")
        return
    write_verified(target, download(lock["bazel"]["url"]), lock["bazel"], executable=True)


def unpopulated_source(relative):
    source = owned_path(relative)
    index = Path(git("-C", relative, "rev-parse", "--git-path", "index"))
    if not index.is_absolute():
        index = source / index
    return (not index.exists()
            and all(entry.name == ".git" for entry in source.iterdir()))


def inspect_sources(lock, *, require_initialized=False):
    records = {}
    for line in (ROOT / "upstream-sources.lock").read_text().splitlines():
        if line and not line.startswith("#"):
            fields = line.split("|")
            if fields[1] in records:
                raise EnvironmentError(f"duplicate source lock entry: {fields[1]}")
            records[fields[1]] = fields
    missing = []
    for relative in lock["source_paths"]:
        path = owned_path(relative)
        record = records[relative]
        revision = record[3]
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise EnvironmentError(f"core source requires a commit, not a moving ref: {relative}")
        index = git("ls-files", "--stage", "--", relative).split()
        if index != ["160000", revision, "0", relative]:
            raise EnvironmentError(f"index gitlink differs from the source lock: {relative}")
        committed = git("rev-parse", f"HEAD:{relative}")
        if committed != revision:
            raise EnvironmentError(f"committed gitlink differs from the source lock: {relative}")
        url = git("config", "-f", ".gitmodules", "--get", f"submodule.{relative}.url")
        if url != record[2]:
            raise EnvironmentError(f"submodule URL differs from the source lock: {relative}")
        try:
            local_url = git("config", "--local", "--get-all", f"submodule.{relative}.url")
        except subprocess.CalledProcessError as error:
            if error.returncode != 1:
                raise
            local_url = None
        if local_url is not None and local_url != url:
            raise EnvironmentError(f"local submodule URL differs from the source lock: {relative}")
        if not (path / ".git").exists():
            if path.exists() and any(path.iterdir()):
                raise EnvironmentError(f"uninitialized source directory is not empty: {relative}")
            missing.append(relative)
            continue
        if Path(git("-C", relative, "rev-parse", "--show-toplevel")) != path:
            raise EnvironmentError(f"not an independent source Git root: {relative}")
        if git("-C", relative, "config", "--get-all", "remote.origin.url") != url:
            raise EnvironmentError(f"source origin URL differs from the source lock: {relative}")
        if unpopulated_source(relative):
            missing.append(relative)
            continue
        if git("-C", relative, "status", "--porcelain", "--untracked-files=all"):
            raise EnvironmentError(f"source has local changes; preserve them before syncing: {relative}")
        if git("-C", relative, "rev-parse", "HEAD") != revision:
            missing.append(relative)
    if require_initialized and missing:
        raise EnvironmentError("missing or mismatched source checkout: " + ", ".join(missing))
    return missing


def sync_sources(lock):
    missing = inspect_sources(lock)
    if missing:
        # Fetch the requested commit directly. A generic submodule update may
        # first fetch the moving default branch when an older commit is absent.
        git("submodule", "init", "--", *missing, capture=False)
        for relative in missing:
            source = owned_path(relative)
            if not (source / ".git").exists():
                git("init", "-q", relative, capture=False)
                url = git("config", "-f", ".gitmodules", "--get", f"submodule.{relative}.url")
                git("-C", relative, "remote", "add", "origin", url, capture=False)
            revision = git("rev-parse", f"HEAD:{relative}")
            git("-C", relative, "config", "remote.origin.promisor", "true", capture=False)
            git("-C", relative, "config", "remote.origin.partialclonefilter", "blob:none", capture=False)
            git("-C", relative, "fetch", "--no-tags", "--depth", "1", "--filter=blob:none",
                "origin", revision, capture=False)
            git("-C", relative, "checkout", "--detach", revision, capture=False)
            git("submodule", "absorbgitdirs", "--", relative, capture=False)
    inspect_sources(lock, require_initialized=True)


def source_bytecode(lock):
    files = []
    for relative in lock["source_paths"]:
        source = owned_path(relative)
        output = git("-C", relative, "ls-files", "--others", "--ignored", "--exclude-standard", "-z")
        for item in output.split("\0"):
            if item and PurePosixPath(item).suffix in {".pyc", ".pyo"}:
                path = owned_path(Path(relative) / item)
                if not stat.S_ISREG(path.lstat().st_mode):
                    raise EnvironmentError(f"unexpected bytecode type: {path}")
                files.append(path)
    return files


def quarantine_bytecode(lock):
    files = source_bytecode(lock)
    if not files:
        return
    backups = owned_path(CACHE / "bytecode")
    backups.mkdir(parents=True, exist_ok=True)
    destination = Path(tempfile.mkdtemp(prefix="backup-", dir=backups))
    for path in files:
        target = destination / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(path, target)
    print(f"quarantined {len(files)} bytecode files: {destination}", flush=True)


@contextmanager
def sync_lock():
    path = owned_path(CACHE / ".sync.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise EnvironmentError("environment lock must be a single-link regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise EnvironmentError("another environment synchronization is running") from error
        yield
    finally:
        os.close(descriptor)


def process_environment(uv):
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("UV_")}
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        environment.pop(key, None)
    cache_name = "docker-uv-cache" if os.environ.get("JAX_ANALYSIS_CONTAINER") == "1" else "host-uv-cache"
    cache = owned_path(CACHE / cache_name)
    environment.update({
        "PATH": f"{uv.parent}:/usr/local/bin:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
        "UV_CACHE_DIR": str(cache), "UV_PYTHON_DOWNLOADS": "never",
        "UV_PROJECT_ENVIRONMENT": str(ROOT / ".venv"),
    })
    if os.environ.get("JAX_ANALYSIS_CONTAINER") == "1":
        # The venv bind mount and the cache have different mount identities.
        environment["UV_LINK_MODE"] = "copy"
    return environment


def verify_workspace(lock, uv, *, strict=False):
    inspect_sources(lock, require_initialized=True)
    bytecode = source_bytecode(lock)
    if bytecode:
        raise EnvironmentError(f"source contains {len(bytecode)} bytecode files; run sync to quarantine them")
    if not matches(uv, lock["uv"]) or not matches(owned_path(lock["bazel"]["path"]), lock["bazel"]):
        raise EnvironmentError("uv or Bazel does not match the environment lock")
    python = owned_path(".venv/bin") / "python"
    # The final venv executable is intentionally a symlink to the OS interpreter.
    # Its parent directories, unlike the executable itself, must be repository-owned.
    if python.resolve() != Path(lock["python"]["path"]):
        raise EnvironmentError("venv must resolve to the locked OS Python path")
    environment = process_environment(uv)
    for arguments in (
        [uv, "lock", "--check", "--offline"],
        [python, "-B", "tools/capture-baseline.py", "--verify"],
        # Historical captures retain their original revisions and patches. Live
        # capture matching is a recording-time gate, not an environment restore.
        [python, "-B", "tools/validate-evidence.py"],
        [python, "-B", "labs/001-jit-cpu/probe.py", "--stage", "run"],
        [python, "-B", "tools/project-status.py", "--check"],
    ):
        run(arguments, env=environment)
    if strict:
        run([python, "-B", "tools/check-jaxlib-build-env.py", "--strict"], env=environment)


def synchronize(lock, *, notebook=False, strict=False):
    with sync_lock():
        if not matches(GIT, lock["git"]):
            raise EnvironmentError("Git binary differs from the lock; use the Docker environment")
        sync_sources(lock)
        uv = ensure_uv(lock)
        ensure_bazel(lock)
        interpreter = Path(lock["python"]["path"])
        if not interpreter.is_file():
            raise EnvironmentError("host Python 3.12.3 is missing; use the Docker environment")
        version = run([interpreter, "-B", "-c", "import sys; print(sys.version.split()[0])"], capture=True)
        if version != lock["python"]["version"]:
            raise EnvironmentError("host Python version differs; use the Docker environment")
        owned_path(".venv")
        arguments = [uv, "sync", "--locked", "--python", interpreter, "--no-python-downloads"]
        if notebook:
            arguments += ["--group", "notebook"]
        run(arguments, env=process_environment(uv))
        quarantine_bytecode(lock)
        verify_workspace(lock, uv, strict=strict)
        print("environment synchronized" + (" [strict container toolchain]" if strict else " [host CPU runtime]"))


def recipe_digest():
    digest = hashlib.sha256()
    for name in ("Dockerfile", "environment.lock.json", "verify-image.py",
                 "install-packages.sh", "fetch-apt-package.sh"):
        digest.update(name.encode() + b"\0" + (ROOT / "env" / name).read_bytes())
    return digest.hexdigest()


def docker_tag():
    return "jax-source-analysis-env:" + recipe_digest()[:20]


def docker_image():
    output = run(["docker", "image", "inspect", docker_tag()], capture=True)
    record = json.loads(output)[0]
    label = record["Config"].get("Labels", {}).get("org.jax-source-analysis.environment")
    if label != recipe_digest() or not re.fullmatch(r"sha256:[0-9a-f]{64}", record["Id"]):
        raise EnvironmentError("Docker image does not match the current locked build recipe")
    return record["Id"]


def build_docker(lock, network="default"):
    uv = ensure_uv(lock)
    directory = owned_path(CACHE / "docker-contexts")
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="build-", dir=directory) as temporary:
        context = Path(temporary)
        ca_identity = lock["docker"]["bootstrap_ca"]
        package = owned_path(CACHE / "ca-certificates.deb")
        if not package.exists():
            write_verified(package, download(ca_identity["url"]), ca_identity)
        if not matches(package, ca_identity):
            raise EnvironmentError("bootstrap CA package hash mismatch")
        filesystem = subprocess.check_output(["dpkg-deb", "--fsys-tarfile", package])
        with tarfile.open(fileobj=io.BytesIO(filesystem)) as archive:
            members = sorted((entry for entry in archive.getmembers()
                              if entry.isfile() and entry.name.startswith("./usr/share/ca-certificates/mozilla/")
                              and entry.name.endswith(".crt")), key=lambda entry: entry.name)
            if not members:
                raise EnvironmentError("CA package contains no Mozilla trust anchors")
            (context / "ca-certificates.crt").write_bytes(
                b"\n".join(archive.extractfile(member).read() for member in members))
        for name in ("Dockerfile", "environment.lock.json", "verify-image.py",
                     "install-packages.sh", "fetch-apt-package.sh"):
            shutil.copyfile(ROOT / "env" / name, context / name)
        shutil.copyfile(uv, context / "uv")
        (context / "uv").chmod(0o755)
        docker = lock["docker"]
        arguments = ["docker", "build", "--platform", lock["platform"], "--pull", "--network", network,
                     "--build-arg", "BASE_IMAGE=" + docker["base_image"],
                     "--build-arg", "APT_SNAPSHOT=" + docker["apt_snapshot"],
                     "--build-arg", "APT_PACKAGES=" + " ".join(docker["apt_packages"]),
                     "--build-arg", "APT_MIRRORS=" + " ".join(docker["package_mirrors"]),
                     "--build-arg", "RECIPE_SHA256=" + recipe_digest(), "--tag", docker_tag()]
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy"):
            if name in os.environ:
                arguments += ["--build-arg", name]
        run([*arguments, context])
    print("built immutable environment image: " + docker_image())


def run_docker(lock, command, *, notebook=False, network="default"):
    with sync_lock():
        sync_sources(lock)
        ensure_bazel(lock)
    image = docker_image()
    venv = owned_path(CACHE / "docker-venv")
    venv.mkdir(parents=True, exist_ok=True)
    owned_path(".venv").mkdir(exist_ok=True)
    if ":" in str(ROOT):
        raise EnvironmentError("Docker bind mount paths cannot contain ':'")
    arguments = ["docker", "run", "--rm", "--platform", lock["platform"],
                 "--user", f"{os.getuid()}:{os.getgid()}",
                 "--volume", f"{ROOT}:{ROOT}", "--volume", f"{venv}:{ROOT / '.venv'}",
                 "--workdir", str(ROOT), "--env", "HOME=/tmp",
                 "--env", "JAX_ANALYSIS_CONTAINER=1"]
    if network == "host":
        arguments += ["--network", "host"]
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy"):
        if name in os.environ:
            arguments += ["--env", name]
    if sys.stdin.isatty() and sys.stdout.isatty():
        arguments += ["--interactive", "--tty"]
    arguments += [image, lock["python"]["path"], "-B", "tools/sync-environment.py", "container-exec"]
    if notebook:
        arguments += ["--notebook"]
    run([*arguments, "--", *command])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    for name in ("sync", "check", "docker", "container-exec"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--notebook", action="store_true")
        if name == "docker":
            sub.add_argument("--network", choices=("default", "host"), default="default")
        if name in ("sync", "check"):
            sub.add_argument("--strict", action="store_true")
        if name in ("docker", "container-exec"):
            sub.add_argument("command", nargs=argparse.REMAINDER)
    build_parser = subparsers.add_parser("docker-build")
    build_parser.add_argument("--network", choices=("default", "host"), default="default")
    args = parser.parse_args()
    try:
        lock = load_lock()
        if args.action == "sync":
            synchronize(lock, notebook=args.notebook, strict=args.strict)
        elif args.action == "check":
            uv = owned_path(CACHE / "bin/uv")
            verify_workspace(lock, uv, strict=args.strict)
        elif args.action == "docker-build":
            build_docker(lock, args.network)
        else:
            command = args.command[1:] if args.command[:1] == ["--"] else args.command
            if args.action == "docker":
                run_docker(lock, command, notebook=args.notebook, network=args.network)
            else:
                if os.environ.get("JAX_ANALYSIS_CONTAINER") != "1":
                    raise EnvironmentError("container-exec is an internal Docker entry point")
                synchronize(lock, notebook=args.notebook, strict=True)
                if command:
                    run(command, env=process_environment(owned_path(CACHE / "bin/uv")))
        return 0
    except (EnvironmentError, OSError, ValueError, KeyError, tarfile.TarError,
            subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        print(f"environment error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
