"""Record hardware/software environment to results/logs/environment_<date>.json."""
import json, platform, os, shutil, subprocess, datetime, importlib
def ver(m):
    try: return importlib.import_module(m).__version__
    except Exception as e: return f"unavailable ({type(e).__name__})"
mem = open("/proc/meminfo").read().split("\n")[0]
cpu = [l for l in open("/proc/cpuinfo") if l.startswith("model name")]
env = {"date": datetime.date.today().isoformat(), "python": platform.python_version(),
       "platform": platform.platform(), "cpu_model": cpu[0].split(":")[1].strip() if cpu else "?",
       "cpu_count": os.cpu_count(), "meminfo_total": mem, "disk_free_GB": round(shutil.disk_usage(".").free/1e9,1),
       "gpu": "none detected", "packages": {m: ver(m) for m in ["numpy","scipy","pandas","matplotlib","jax","jaxlib","torch"]}}
try: env["git_commit"] = subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
except Exception: env["git_commit"] = "none"
os.makedirs("results/logs", exist_ok=True)
p = f"results/logs/environment_{env['date']}.json"; json.dump(env, open(p,"w"), indent=2); print(json.dumps(env, indent=2))
