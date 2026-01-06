"""
vmd_hrv_offline.py (optimized)

Offline HRV bileşen analizi:
- rr_clean klasöründeki *_clean.csv dosyalarından RR okur
- RR -> 2 Hz uniform HRV(t) sinyaline çevirir
- Üç yöntemden biriyle HF / LF / VLF / ULF bileşenlerini çıkarır:
    * klasik VMD      (--method vmd)
    * adaptif VMD     (--method avmd)  [HRV-odaklı, band coverage + energy loss]
    * VMDon benzeri   (--method vmdon) [sliding-window, offline emülasyon]

İyileştirmeler (AVMD tarafı):
- K seçimi sadece "reconstruction error" ile değil, HRV band kapsaması ile birlikte yapılır
  (HF/LF/VLF en az birer moda sahip olana kadar K artar; ardından energy loss eşiği kontrol edilir).
- Mode band ataması Welch "peak bin" yerine (varsa) VMD'nin omega merkez frekansları ile yapılır.
- 30 dk pencerede ULF'nin zayıf/boş olması normal kabul edilir.
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Literal, Optional, Iterable, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline
from scipy.signal import welch, hilbert, savgol_filter

from vmdpy import VMD


# ----------------- PATH & GLOBAL CONFIG ----------------- #

ROOT_DIR = Path(__file__).resolve().parent
RR_DIR = ROOT_DIR / "data" / "processed" / "rr_clean"

# Çıktı klasörleri
PLOT_DIR = ROOT_DIR / "data" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH = PLOT_DIR / "band_summary.csv"

FS_RESAMPLE: float = 2.0  # Hz

VMD_ALPHA: float = 2000.0
VMD_TAU: float = 0.0
VMD_INIT: int = 1
VMD_TOL: float = 1e-7

HRV_BANDS = {
    "ULF": (0.0,    0.0047),
    "VLF": (0.0047, 0.0300),
    "LF":  (0.0300, 0.1400),
    "HF":  (0.1100, 0.4000),
}

REQUIRED_BANDS: Tuple[str, ...] = ("HF", "LF", "VLF")

VMDON_WINDOW_S: Dict[str, float] = {
    "HF": 6.5,
    "LF": 25.0,
    "VLF": 303.0,
    "ULF": 1800.0,
}


# ----------------- LOAD & RESAMPLE ----------------- #

def load_rr_sec_from_clean(subject_code: str) -> np.ndarray:
    """Load cleaned RR series (seconds) for a subject."""
    try:
        code_int = int(subject_code)
        candidates = [
            RR_DIR / f"{code_int}_clean.csv",
            RR_DIR / f"{code_int:03d}_clean.csv",
        ]
    except ValueError:
        candidates = [RR_DIR / f"{subject_code}_clean.csv"]

    csv_path = None    # type: ignore
    for c in candidates:
        if c.exists():
            csv_path = c
            break

    if csv_path is None:
        raise FileNotFoundError(f"No RR file found for subject={subject_code} in {RR_DIR}")

    print(f"[INFO] Using RR file: {csv_path}")
    df = pd.read_csv(csv_path)

    rr_col_candidates = ["rr_sec", "rr_s", "rr", "rr_ms"]
    rr_col = None
    for col in rr_col_candidates:
        if col in df.columns:
            rr_col = col
            break
    if rr_col is None:
        raise ValueError(f"Could not find RR column in {csv_path}. Available: {list(df.columns)}")

    rr = df[rr_col].to_numpy(dtype=float)

    if rr_col.lower().endswith("ms"):
        rr_sec = rr / 1000.0
    else:
        rr_sec = rr

    rr_sec = rr_sec[np.isfinite(rr_sec)]
    rr_sec = rr_sec[rr_sec > 0.1]
    return rr_sec


def rr_to_uniform_hrv(
    rr_sec: np.ndarray,
    fs: float = FS_RESAMPLE,
    max_minutes: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """RR (s) -> uniform HRV(t) (fs Hz) via cubic spline."""
    rr_sec = np.asarray(rr_sec, dtype=float)
    if rr_sec.size == 0:
        return np.asarray([]), np.asarray([])

    t_rr = np.cumsum(rr_sec)
    t_rr = t_rr - t_rr[0]

    if max_minutes is not None:
        max_t = max_minutes * 60.0
        mask = t_rr <= max_t
        if mask.sum() >= 4:
            rr_sec = rr_sec[mask]
            t_rr = t_rr[mask]

    total_dur = float(t_rr[-1])
    dt = 1.0 / fs
    t_uniform = np.arange(0.0, total_dur, dt)

    if t_uniform.size < 4:
        hrv = np.interp(t_uniform, t_rr, rr_sec)
        return t_uniform, hrv

    cs = CubicSpline(t_rr, rr_sec)
    hrv = cs(t_uniform)
    return t_uniform, hrv


# ----------------- PREPROCESS ----------------- #

def detrend_signal(x: np.ndarray, mode: Literal["none", "mean", "linear"] = "mean") -> np.ndarray:
    """Detrend HRV signal (none/mean/linear)."""
    x = np.asarray(x, dtype=float)
    if mode == "none":
        return x
    if mode == "mean":
        return x - np.mean(x)
    if mode == "linear":
        t = np.arange(x.size, dtype=float)
        p = np.polyfit(t, x, 1)
        return x - np.polyval(p, t)
    raise ValueError(f"Unknown detrend mode: {mode}")


# ----------------- VMD helpers ----------------- #

def run_vmd(signal: np.ndarray, K: int, alpha: float, dc: int) -> Tuple[np.ndarray, Any]:
    """Run VMD and return modes + omega."""
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1 or signal.size < 10:
        raise ValueError("Signal too short for VMD.")

    u, u_hat, omega = VMD(signal, alpha, VMD_TAU, K, dc, VMD_INIT, VMD_TOL)
    return u, omega


def _omega_to_hz(omega: np.ndarray, fs: float) -> np.ndarray:
    """
    Convert VMD omega output to Hz in a robust way.
    Common cases:
      - cycles/sample in [0..0.5]  => Hz = omega * fs
      - rad/sample   in [0..pi]    => Hz = omega * fs / (2*pi)
    """
    om = np.asarray(omega, dtype=float)
    if om.ndim == 2:
        om = om[-1]  # final iteration row
    om = np.asarray(om).reshape(-1)

    if om.size == 0 or not np.isfinite(np.nanmax(om)):
        return om

    max_om = float(np.nanmax(om))
    if max_om <= 0.5 + 1e-6:
        return om * fs
    if max_om <= np.pi + 1e-6:
        return om * fs / (2.0 * np.pi)
    return om  # assume already Hz


def compute_mode_peak_freq(mode: np.ndarray, fs: float, nperseg: int = 1024) -> float:
    """Welch PSD peak frequency (fallback)."""
    mode = np.asarray(mode, dtype=float)
    if mode.size < 8:
        return np.nan
    nps = min(nperseg, mode.size)
    f, Pxx = welch(mode, fs=fs, nperseg=nps)
    if f.size == 0:
        return np.nan
    return float(f[int(np.argmax(Pxx))])


def assign_modes_to_bands(
    modes: np.ndarray,
    fs: float,
    omega: Optional[Any] = None,
    use_omega: bool = True
) -> Dict[int, str]:
    """Assign each mode to a HRV band using omega (preferred) or Welch peak freq."""
    mode_band: Dict[int, str] = {}

    omega_hz: Optional[np.ndarray] = None
    if use_omega and omega is not None:
        try:
            omega_hz = _omega_to_hz(np.asarray(omega), fs=fs)
            if omega_hz.size != modes.shape[0]:
                omega_hz = None
        except Exception:
            omega_hz = None

    # IMPORTANT: Use explicit band order to avoid VLF/LF overlap issues.
    band_order = ["ULF", "VLF", "LF", "HF"]

    for i in range(modes.shape[0]):
        f_i = float(omega_hz[i]) if omega_hz is not None else compute_mode_peak_freq(modes[i], fs=fs)

        band_name = "?"
        if np.isfinite(f_i):
            for name in band_order:
                fmin, fmax = HRV_BANDS[name]

                # Half-open interval helps with boundary overlaps (e.g., 0.029-0.031 region).
                if fmin <= f_i < fmax:
                    band_name = name
                    break

            # If still unmatched and it's exactly the top HF edge, include it.
            if band_name == "?" and "HF" in HRV_BANDS:
                hf_min, hf_max = HRV_BANDS["HF"]
                if np.isfinite(hf_max) and np.isclose(f_i, hf_max):
                    band_name = "HF"

        mode_band[i] = band_name
        print(f"[INFO] Mode {i}: f={f_i:.5f} Hz -> {band_name}")

    return mode_band


def reconstruct_components(modes: np.ndarray, mode_band: Dict[int, str]) -> Dict[str, np.ndarray]:
    """Sum modes per band."""
    n = modes.shape[1]
    comps: Dict[str, np.ndarray] = {}
    for band in HRV_BANDS.keys():
        idx = [i for i, b in mode_band.items() if b == band]
        comps[band] = np.sum(modes[idx, :], axis=0) if idx else np.zeros(n, dtype=float)
    return comps


# ----------------- HRV-aware AVMD ----------------- #

def _coverage_score(mode_band: Dict[int, str], required: Iterable[str]) -> int:
    present = set(mode_band.values())
    return sum(1 for b in required if b in present)


def run_avmd_hrv(
    signal: np.ndarray,
    fs: float,
    alpha: float,
    kmin: int,
    kmax: int,
    energy_loss: float,
    required_bands: Tuple[str, ...],
    use_omega: bool,
    dc: int,
) -> Tuple[np.ndarray, Any, int, Dict[int, str]]:
    """
    HRV-focused AVMD:
    Select smallest K that satisfies:
      coverage(required_bands) == all  AND  reconstruction loss <= energy_loss
    Otherwise select best by (coverage desc, loss asc).
    """
    x = np.asarray(signal, dtype=float)

    best = {"K": None, "loss": float("inf"), "cov": -1, "modes": None, "omega": None, "band": None}

    for K in range(kmin, kmax + 1):
        modes, omega = run_vmd(x, K=K, alpha=alpha, dc=dc)
        rec = np.sum(modes, axis=0)

        minlen = min(len(x), len(rec))
        loss = float(np.linalg.norm(x[:minlen] - rec[:minlen]) / np.linalg.norm(x[:minlen]))

        band = assign_modes_to_bands(modes, fs=fs, omega=omega, use_omega=use_omega)
        cov = _coverage_score(band, required_bands)

        print(f"[AVMD-HRV] K={K}, loss={loss:.6f}, coverage={cov}/{len(required_bands)}")

        if cov == len(required_bands) and loss <= energy_loss:
            print(f"[AVMD-HRV] Selected K={K} (meets coverage + loss)")
            return modes, omega, K, band

        better = (cov > best["cov"]) or (cov == best["cov"] and loss < best["loss"])
        if better:
            best.update({"K": K, "loss": loss, "cov": cov, "modes": modes, "omega": omega, "band": band})

    print(f"[AVMD-HRV] Selected K={best['K']} (best coverage/loss fallback)")
    return best["modes"], best["omega"], int(best["K"]), best["band"]


# ----------------- VMDON (offline emülasyon) ----------------- #

def _sliding_component(signal: np.ndarray, fs: float, window_s: float, use_vmd: bool, alpha: float, stride: int) -> np.ndarray:
    """Compute one sliding-window component (HF/LF/VLF/ULF)."""
    x = np.asarray(signal, dtype=float)
    N = x.size
    W = int(round(window_s * fs))
    if W < 4 or W >= N:
        raise ValueError(f"Window too short/long: W={W}, N={N}")

    acc = np.zeros(N, dtype=float)
    cnt = np.zeros(N, dtype=float)
    t_local = np.arange(W)

    for start in range(0, N - W + 1, stride):
        w = x[start:start + W]
        p = np.polyfit(t_local, w, 1)
        trend = np.polyval(p, t_local)
        w_detr = w - trend

        if use_vmd:
            u, _ = run_vmd(w_detr, K=1, alpha=alpha, dc=0)
            comp_win = u[0]
        else:
            comp_win = trend

        acc[start:start + W] += comp_win
        cnt[start:start + W] += 1.0

    out = np.zeros(N, dtype=float)
    m = cnt > 0
    out[m] = acc[m] / cnt[m]
    return out


def vmdon_offline(hrv: np.ndarray, fs: float, stride_hf: int = 1, stride_lf: int = 2, stride_vlf: int = 10, stride_ulf: int = 60) -> Dict[str, np.ndarray]:
    """VMDon-like offline decomposition with sliding windows."""
    x = np.asarray(hrv, dtype=float)

    hf = _sliding_component(x, fs, VMDON_WINDOW_S["HF"], True, VMD_ALPHA, stride_hf)
    r = x - hf

    lf = _sliding_component(r, fs, VMDON_WINDOW_S["LF"], True, VMD_ALPHA, stride_lf)
    r = r - lf

    vlf = _sliding_component(r, fs, VMDON_WINDOW_S["VLF"], True, VMD_ALPHA, stride_vlf)
    r = r - vlf

    ulf = _sliding_component(r, fs, VMDON_WINDOW_S["ULF"], False, VMD_ALPHA, stride_ulf)

    return {"HF": hf, "LF": lf, "VLF": vlf, "ULF": ulf}


# ----------------- HILBERT AM/FM ----------------- #

def hilbert_am_fm(x: np.ndarray, fs: float, sg_window_s: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Compute smoothed Hilbert amplitude and instantaneous frequency."""
    x = np.asarray(x, dtype=float)
    analytic = hilbert(x)
    amp = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))

    if sg_window_s is None:
        sg_window_s = 1.0

    win_len = int(round(sg_window_s * fs))
    if win_len % 2 == 0:
        win_len += 1
    win_len = max(win_len, 5)

    amp_s = savgol_filter(amp, win_len, polyorder=2)
    ph_s = savgol_filter(phase, win_len, polyorder=2)

    dph = np.gradient(ph_s) * fs
    freq = dph / (2.0 * np.pi)
    return amp_s, freq


# ----------------- PLOT & SUMMARY ----------------- #

def plot_components(
    t: np.ndarray,
    hrv: np.ndarray,
    comps: Dict[str, np.ndarray],
    subject: str,
    method: str,
    save: bool = True,
) -> None:
    """Plot HRV components and optionally save to PNG."""
    fig, axes = plt.subplots(5, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(t, hrv, linewidth=0.8)
    axes[0].set_title(f"Subject {subject} - Uniform HRV(t) [{method}]")
    axes[0].set_ylabel("RR (s)")
    axes[0].grid(True, alpha=0.3)

    for ax, band in zip(axes[1:], ["HF", "LF", "VLF", "ULF"]):
        ax.plot(t, comps[band], linewidth=0.8)
        ax.set_ylabel(band)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()

    if save:
        out_dir = PLOT_DIR / method
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{subject}_{method}_components.png"
        plt.savefig(out_path, dpi=200)
        print(f"[INFO] Saved plot to {out_path}")
        plt.close(fig)
    else:
        plt.show()


def print_band_summary(comps: Dict[str, np.ndarray], subject: str, method: str) -> None:
    """Print and append band stats to CSV."""
    print("\n[SUMMARY] Band powers (variance) & basic stats:")
    rows = []
    for band, x in comps.items():
        x = np.asarray(x, dtype=float)
        mean = float(np.mean(x))
        var = float(np.var(x))
        var_demean = float(np.var(x - mean))
        print(f"  {band}: var={var:.6e}, mean={mean:.4f}, var_demean={var_demean:.6e}")

        rows.append(
            {
                "subject": subject,
                "method": method,
                "band": band,
                "var": var,
                "mean": mean,
                "var_demean": var_demean,
            }
        )

    df = pd.DataFrame(rows)
    header = not SUMMARY_PATH.exists()
    df.to_csv(SUMMARY_PATH, mode="a", index=False, header=header)
    print(f"[INFO] Appended band summary to {SUMMARY_PATH}")


# ----------------- MAIN ----------------- #

def run_for_subject(
    subject: str,
    method: Literal["vmd", "avmd", "vmdon"],
    max_minutes: Optional[float],
    detrend: Literal["none", "mean", "linear"],
    kmin: int,
    kmax: int,
    energy_loss: float,
    use_omega: bool,
    dc: int,
) -> None:
    """Main offline pipeline for one subject."""
    rr = load_rr_sec_from_clean(subject)
    print(f"[INFO] Loaded {rr.size} RR intervals (sec) for subject={subject}")
    print(f"[INFO] Approx duration: {rr.sum()/60.0:.1f} minutes")

    t, hrv = rr_to_uniform_hrv(rr, fs=FS_RESAMPLE, max_minutes=max_minutes)
    print(f"[INFO] Uniform HRV length: {hrv.size} samples ({t[-1]/60.0:.1f} min) at {FS_RESAMPLE} Hz")

    hrv_p = detrend_signal(hrv, mode=detrend)

    if method == "avmd":
        modes, omega, K, band = run_avmd_hrv(
            hrv_p, FS_RESAMPLE,
            alpha=VMD_ALPHA,
            kmin=kmin, kmax=kmax,
            energy_loss=energy_loss,
            required_bands=REQUIRED_BANDS,
            use_omega=use_omega,
            dc=dc,
        )
        print(f"[INFO] AVMD-HRV modes shape: {modes.shape} (K={K})")
        comps = reconstruct_components(modes, band)

    elif method == "vmd":
        K = max(4, kmin)
        modes, omega = run_vmd(hrv_p, K=K, alpha=VMD_ALPHA, dc=dc)
        band = assign_modes_to_bands(modes, FS_RESAMPLE, omega=omega, use_omega=use_omega)
        comps = reconstruct_components(modes, band)

    elif method == "vmdon":
        comps = vmdon_offline(hrv_p, FS_RESAMPLE)

    else:
        raise ValueError("Unknown method")

    # PNG + CSV çıktıları
    plot_components(t[:comps["HF"].size], hrv_p[:comps["HF"].size], comps, subject, method)
    print_band_summary(comps, subject, method)

    print("\n[INFO] Example AM/FM for HF band (first 10 minutes segment).")
    mask = t <= 10*60.0
    hf_seg = comps["HF"][mask[:comps["HF"].size]]
    amp, freq = hilbert_am_fm(hf_seg, FS_RESAMPLE, sg_window_s=2.0)
    print(f"  HF AM median: {np.median(amp):.4f}, HF FM median: {np.median(freq):.4f} Hz")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--subject", required=True)
    p.add_argument("--method", choices=["vmd", "avmd", "vmdon"], default="avmd")
    p.add_argument("--max-minutes", type=float, default=30.0)
    p.add_argument("--detrend", choices=["none", "mean", "linear"], default="mean")
    p.add_argument("--kmin", type=int, default=4)
    p.add_argument("--kmax", type=int, default=12)
    p.add_argument("--energy-loss", type=float, default=0.01)
    p.add_argument("--no-omega", action="store_true")
    p.add_argument("--dc", type=int, choices=[0, 1], default=0)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run_for_subject(
        a.subject,
        method=a.method,
        max_minutes=a.max_minutes,
        detrend=a.detrend,
        kmin=a.kmin,
        kmax=a.kmax,
        energy_loss=a.energy_loss,
        use_omega=(not a.no_omega),
        dc=a.dc,
    )

# Örnek:
# python vmd_hrv_offline.py --subject 003 --method avmd --max-minutes 30 --detrend mean --kmin 4 --kmax 12 --energy-loss 0.05
