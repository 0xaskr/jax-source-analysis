#!/usr/bin/env python3
"""Install, check, or run the locked diagram environment without root privileges.

Ubuntu 24.04 amd64 and Python 3.12 are required. Packages are extracted under
artifacts/environment/diagrams; the system and the JAX virtualenv are untouched.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
from html import escape
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "env/diagrams.lock.json"
CACHE = ROOT / "artifacts/environment/diagrams"
SETUP = "python3 -B tools/diagram_environment.py sync"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_lock():
    lock = json.loads(LOCK.read_text())
    if lock["schema_version"] != 1 or lock["platform"] != "ubuntu-24.04-amd64":
        raise RuntimeError("Unsupported diagram environment lock")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("The diagram environment requires Linux amd64")
    if f"{sys.version_info.major}.{sys.version_info.minor}" != lock["python_abi"]:
        raise RuntimeError("Use Python " + lock["python_abi"] + " for the locked drawing bindings")
    for package in lock["packages"]:
        if not package["url"].startswith("https://archive.ubuntu.com/ubuntu/pool/"):
            raise RuntimeError("Diagram packages must come from the official Ubuntu archive")
        if len(package["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in package["sha256"]):
            raise RuntimeError("Invalid diagram package checksum")
    return lock


def bundle():
    return CACHE / digest(LOCK)[:20]


def matches(path, package):
    return path.is_file() and path.stat().st_size == package["size_bytes"] and digest(path) == package["sha256"]


def download(package):
    directory = CACHE / "downloads"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (package["sha256"] + ".deb")
    if matches(path, package):
        return path
    filename = package["url"].rsplit("/", 1)[1]
    system_cache = Path("/var/cache/apt/archives") / filename
    if matches(system_cache, package):
        shutil.copyfile(system_cache, path)
        return path
    print(f'Download {package["name"]}={package["version"]}', flush=True)
    with tempfile.NamedTemporaryFile(dir=directory, delete=False) as temporary:
        pending = Path(temporary.name)
        try:
            with urllib.request.urlopen(package["url"], timeout=45) as response:
                shutil.copyfileobj(response, temporary)
            temporary.close()
            if not matches(pending, package):
                raise RuntimeError("Package SHA-256 or size mismatch: " + filename)
            os.replace(pending, path)
        finally:
            pending.unlink(missing_ok=True)
    return path


def verify_bundle():
    load_lock()
    target = bundle()
    manifest_path = target / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("Drawing dependencies are not installed. Run: " + SETUP)
    manifest = json.loads(manifest_path.read_text())
    if manifest["lock_sha256"] != digest(LOCK):
        raise RuntimeError("Diagram lock changed. Run: " + SETUP)
    for relative, expected in manifest["files"].items():
        path = target / "root" / relative
        if not path.is_file() or digest(path) != expected:
            raise RuntimeError("Drawing dependency changed: " + relative)
    return target


def synchronize():
    lock = load_lock()
    CACHE.mkdir(parents=True, exist_ok=True)
    target = bundle()
    if target.exists():
        verify_bundle()
        print("Drawing packages already match the lock")
        return
    with tempfile.TemporaryDirectory(prefix="install-", dir=CACHE) as temporary:
        staging = Path(temporary)
        root = staging / "root"
        root.mkdir()
        for package in lock["packages"]:
            subprocess.run(["dpkg-deb", "--extract", str(download(package)), str(root)], check=True)
        files = {p.relative_to(root).as_posix(): digest(p)
                 for p in sorted(root.rglob("*")) if p.is_file() and not p.is_symlink()}
        (staging / "manifest.json").write_text(json.dumps({
            "lock_sha256": digest(LOCK), "files": files,
        }, indent=2) + "\n")
        # Rename only a newly assembled bundle; never overwrite an existing one.
        staging.rename(target)
    print("Drawing dependencies installed in " + str(target.relative_to(ROOT)))


def environment():
    load_lock()
    target = bundle()
    if not (target / "manifest.json").is_file():
        raise RuntimeError("Drawing dependencies are not installed. Run: " + SETUP)
    root = target / "root"
    font_cache = target / "font-cache"
    font_cache.mkdir(exist_ok=True)
    config = target / "fonts.conf"
    xml = ('<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "urn:fontconfig:fonts.dtd">'
           '<fontconfig><dir>' + escape(str(root / "usr/share/fonts/opentype/noto")) + '</dir>'
           '<cachedir>' + escape(str(font_cache)) + '</cachedir>'
           '<alias><family>sans-serif</family><prefer><family>Noto Sans CJK SC</family></prefer></alias>'
           '</fontconfig>\n')
    if not config.exists() or config.read_text() != xml:
        config.write_text(xml)
    env = dict(os.environ)
    env.update({
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(root / "usr/lib/python3/dist-packages"),
        "GI_TYPELIB_PATH": str(root / "usr/lib/x86_64-linux-gnu/girepository-1.0"),
        "LD_LIBRARY_PATH": str(root / "usr/lib/x86_64-linux-gnu") + ":" + str(root / "lib/x86_64-linux-gnu"),
        "FONTCONFIG_FILE": str(config), "FONTCONFIG_PATH": str(target),
        "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
    })
    return env


@lru_cache(maxsize=1)
def configure():
    """Allow existing render commands to discover the repository-local bindings."""
    env = environment()
    if os.environ.get("LD_LIBRARY_PATH") != env["LD_LIBRARY_PATH"]:
        # The ELF loader reads LD_LIBRARY_PATH at process start. Updating it
        # after importing GI can load two copies of GLib/cairo and deadlock.
        script = Path(sys.argv[0]).resolve()
        if script.parent == ROOT / "tools" and script.name.startswith("render_"):
            os.execve(sys.executable, [sys.executable, "-B", str(script), *sys.argv[1:]], env)
        raise RuntimeError("Start the renderer with: python3 -B tools/diagram_environment.py run SCRIPT [ARGS]")
    sys.path.insert(0, env["PYTHONPATH"])
    for name in ("GI_TYPELIB_PATH", "FONTCONFIG_FILE", "FONTCONFIG_PATH"):
        os.environ[name] = env[name]
    import gi
    # PYTHONPATH, GI_TYPELIB_PATH and the library path now identify one bundle
    # before either the Python bindings or any rendering library is loaded.


@lru_cache(maxsize=1)
def pango():
    configure()
    import gi
    gi.require_version("Pango", "1.0")
    gi.require_version("PangoCairo", "1.0")
    from gi.repository import Pango, PangoCairo
    return Pango, PangoCairo


def preview_bindings():
    configure()
    import gi
    gi.require_version("Rsvg", "2.0")
    gi.require_foreign("cairo")
    import cairo
    from gi.repository import Rsvg
    return Rsvg, cairo


def smoke_check():
    Pango, PangoCairo = pango()
    Rsvg, cairo = preview_bindings()
    context = PangoCairo.FontMap.get_default().create_context()
    for weight in (Pango.Weight.NORMAL, Pango.Weight.BOLD):
        font = Pango.FontDescription()
        font.set_family(load_lock()["font_family"])
        font.set_weight(weight)
        font.set_absolute_size(24 * Pango.SCALE)
        layout = Pango.Layout.new(context)
        layout.set_font_description(font)
        layout.set_text("内层程序 / Pallas → 外层 custom_call", -1)
        if layout.get_unknown_glyphs_count():
            raise RuntimeError("The locked font is missing diagram glyphs")
    svg = b'<svg xmlns="http://www.w3.org/2000/svg" width="100" height="60"><rect width="100" height="60" fill="#2866a4"/></svg>'
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 100, 60)
    rect = Rsvg.Rectangle()
    rect.x = rect.y = 0
    rect.width, rect.height = 100, 60
    if not Rsvg.Handle.new_from_data(svg).render_document(cairo.Context(surface), rect):
        raise RuntimeError("SVG preview render failed")
    with tempfile.TemporaryDirectory(prefix="diagram-smoke-") as temporary:
        png = Path(temporary) / "preview.png"
        surface.write_to_png(str(png))
        if not png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
            raise RuntimeError("PNG encoding failed")
    print(f"PASS: Pango {Pango.version_string()}, cairo {cairo.cairo_version_string()}, "
          f"Rsvg {Rsvg.MAJOR_VERSION}.{Rsvg.MINOR_VERSION}.{Rsvg.MICRO_VERSION}; "
          "Noto Sans CJK SC regular/bold; SVG → PNG")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("sync", help="Download checksum-locked packages and extract them locally")
    check = sub.add_parser("check", help="Check installed file hashes, Chinese fonts, and PNG rendering")
    check.add_argument("--smoke", action="store_true", help=argparse.SUPPRESS)
    run = sub.add_parser("run", help="Run a Python renderer with locked library and font paths")
    run.add_argument("script", type=Path)
    run.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        if args.action == "check" and args.smoke:
            smoke_check()
            return 0
        if args.action == "sync":
            synchronize()
        if args.action in ("sync", "check"):
            verify_bundle()
            return subprocess.run([sys.executable, "-B", __file__, "check", "--smoke"], env=environment()).returncode
        verify_bundle()
        return subprocess.run([sys.executable, "-B", str(args.script), *args.arguments], env=environment()).returncode
    except (OSError, RuntimeError, ValueError, ImportError, subprocess.CalledProcessError) as exc:
        print("diagram environment error: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
