import json, os, platform, sys, importlib, subprocess, shlex

def sh(cmd: str) -> str:
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True).strip()
    except Exception:
        return ""

def nvidia_smi_cuda_driver():
    txt = sh("nvidia-smi")
    cuda, drv = None, None
    if txt:
        for line in txt.splitlines():
            if "CUDA Version" in line and "Driver Version" in line:
                # Example: | NVIDIA-SMI 535.183.01   Driver Version: 535.183.01   CUDA Version: 12.2 |
                # Fallback parse without strict formatting assumptions
                try:
                    parts = line.replace("|", " ").split()
                    if "Version:" in parts:
                        # capture tokens following markers
                        for i, tok in enumerate(parts):
                            if tok == "Version:" and i>0 and parts[i-1] == "Driver":
                                drv = parts[i+1]
                            if tok == "Version:" and i>0 and parts[i-1] == "CUDA":
                                cuda = parts[i+1]
                except Exception:
                    pass
    return cuda, drv

def read_os_release() -> dict:
    d = {}
    try:
        with open("/etc/os-release", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                v = v.strip().strip('"').strip("'")
                d[k] = v
    except Exception:
        pass
    return d

def read_meminfo() -> dict:
    d = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    k, v = line.split(":", 1)
                    d[k.strip()] = v.strip()
    except Exception:
        pass
    return d

def cpu_info() -> dict:
    info = {
        "arch": platform.machine(),
        "count_logical": os.cpu_count(),
    }
    # Model/vendor from /proc/cpuinfo
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            model_name = None
            vendor_id = None
            phys = set()
            cores = set()
            for line in f:
                if line.startswith("model name") and model_name is None:
                    model_name = line.split(":",1)[1].strip()
                if line.startswith("vendor_id") and vendor_id is None:
                    vendor_id = line.split(":",1)[1].strip()
                if line.startswith("physical id"):
                    phys.add(line.split(":",1)[1].strip())
                if line.startswith("core id"):
                    cores.add(line.split(":",1)[1].strip())
            if model_name:
                info["model_name"] = model_name
            if vendor_id:
                info["vendor_id"] = vendor_id
            if phys:
                info["sockets"] = len(phys)
            if cores:
                info["unique_core_ids"] = len(cores)
    except Exception:
        pass

    # Try lscpu JSON if available for richer details
    out = sh("lscpu --json")
    if out:
        try:
            j = json.loads(out)
            # Flatten key-value list under key 'lscpu'
            kv = {e.get("field"," ").strip().strip(":"): e.get("data") for e in j.get("lscpu", [])}
            for k in ["Model name", "Vendor ID", "CPU(s)", "Core(s) per socket", "Thread(s) per core", "Socket(s)", "L3 cache", "Architecture"]:
                if k in kv:
                    info[k.lower().replace(" ", "_").replace("(", "").replace(")", "")] = kv[k]
        except Exception:
            pass
    else:
        # Plain text fallback: keep it minimal
        ls = sh("lscpu")
        if ls:
            info["lscpu"] = ls
    return info

def gpus_info() -> list:
    q = (
        "index,name,uuid,compute_cap,driver_version,"
        "temperature.gpu,utilization.gpu,utilization.memory,"
        "memory.total,memory.used,memory.free,"
        "clocks.sm,clocks.mem,power.draw,power.limit"
    )
    out = sh(f"nvidia-smi --query-gpu={shlex.quote(q)} --format=csv,noheader,nounits")
    gpus = []
    if out:
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 15:
                (idx, name, uuid, cc, drv, temp, util_g, util_m, mem_t, mem_u, mem_f, clk_sm, clk_mem, pwr_d, pwr_l) = parts[:15]
                gpus.append({
                    "index": int(idx),
                    "name": name,
                    "uuid": uuid,
                    "compute_cap": cc,
                    "driver_version": drv,
                    "temperature_c": _to_num(temp),
                    "util_gpu_pct": _to_num(util_g),
                    "util_mem_pct": _to_num(util_m),
                    "mem_total_mb": _to_num(mem_t),
                    "mem_used_mb": _to_num(mem_u),
                    "mem_free_mb": _to_num(mem_f),
                    "clock_sm_mhz": _to_num(clk_sm),
                    "clock_mem_mhz": _to_num(clk_mem),
                    "power_draw_w": _to_num(pwr_d),
                    "power_limit_w": _to_num(pwr_l),
                })
    return sorted(gpus, key=lambda d: d.get("index", 0))

def _to_num(x):
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "":
            return None
        if "." in s:
            return float(s)
        return int(s)
    except Exception:
        return None

info = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    # System basics
    "system": {
        "system": platform.system(),
        "machine": platform.machine(),
        "release": platform.release(),
        "version": platform.version(),
        "kernel": platform.uname().release,
        "os_release": read_os_release(),
    },
    # Packages of interest
    "packages": {
        "torch": None,
        "vllm": None,
        "transformers": None,
    },
    # CUDA/driver
    "cuda": {
        "nvidia_smi_cuda": None,
        "nvidia_driver": None,
        "torch_cuda": None,
        "cudnn": None,
    },
    # CPU and memory
    "cpu": {},
    "memory": {},
}

for m in ("torch", "vllm", "transformers"):
    try:
        info["packages"][m] = importlib.import_module(m).__version__
    except Exception:
        info["packages"][m] = None

# CUDA versions from nvidia-smi and torch
cuda_ver, drv_ver = nvidia_smi_cuda_driver()
info["cuda"]["nvidia_smi_cuda"] = cuda_ver
info["cuda"]["nvidia_driver"] = drv_ver
try:
    import torch  # noqa: E402
    info["cuda"]["torch_cuda"] = getattr(torch.version, "cuda", None)
    try:
        import torch.backends.cudnn as cudnn
        info["cuda"]["cudnn"] = getattr(cudnn, "version", lambda: None)()
    except Exception:
        pass
    try:
        info["cuda"]["avail"] = bool(torch.cuda.is_available())
        info["cuda"]["device_count"] = int(torch.cuda.device_count())
    except Exception:
        pass
except Exception:
    pass

# CPU and memory details
info["cpu"] = cpu_info()
mem = read_meminfo()
def _parse_kb(v):
    try:
        # format: '491693112 kB'
        return int(v.split()[0])
    except Exception:
        return None
info["memory"] = {
    "mem_total_kb": _parse_kb(mem.get("MemTotal", "")),
    "mem_free_kb": _parse_kb(mem.get("MemFree", "")),
    "mem_available_kb": _parse_kb(mem.get("MemAvailable", "")),
    "swap_total_kb": _parse_kb(mem.get("SwapTotal", "")),
}

# GPU detailed list
gpus = gpus_info()

os.makedirs("results", exist_ok=True)
with open("results/env.json", "w", encoding="utf-8") as f:
    json.dump(info, f, indent=2)
with open("results/gpu.json", "w", encoding="utf-8") as f:
    json.dump({"gpus": gpus}, f, indent=2)
with open("results/cpu.json", "w", encoding="utf-8") as f:
    json.dump(info.get("cpu", {}), f, indent=2)