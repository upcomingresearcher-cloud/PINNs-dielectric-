"""Provenance recorded on every database row."""
import datetime, importlib, json, os, platform, subprocess, functools

@functools.lru_cache(None)
def git_commit():
    """Short git commit hash, suffixed '-dirty' if src, configs or scripts have uncommitted changes; 'none' outside git."""
    try:
        h = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD", "--", "src", "configs", "scripts"]) != 0
        return h[:12] + ("-dirty" if dirty else "")
    except Exception:
        return "none"

@functools.lru_cache(None)
def software_env():
    """Versions of Python and the numerical libraries used."""
    v = {"python": platform.python_version()}
    for m in ("numpy", "scipy", "jax", "optax"):
        try: v[m] = importlib.import_module(m).__version__
        except Exception: v[m] = "na"
    return json.dumps(v, sort_keys=True)

@functools.lru_cache(None)
def hardware():
    """CPU model, core count and GPU note of the machine."""
    cpu = "?"
    try: cpu = next(l.split(":")[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except Exception: pass
    return f"{cpu}; cores={os.cpu_count()}; gpu=none"

def now():
    """Current local time as an ISO-8601 string (seconds)."""
    return datetime.datetime.now().isoformat(timespec="seconds")
