"""HI power-to-noise ratios and power-spectrum detection S/N for 9.16.

Use the existing signal, thermal-noise model and cylindrical mode selection.
No parameter Fisher matrices, voxel variances or conditional Noise tables are
recomputed. Wavenumbers are comoving Mpc^-1 and temperatures are mK.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import inspect
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import forecast_916 as forecast


LABELS = {
    "ska_mid1_interferom": "SKA-MID", "chime_icyl": "CHIME",
    "hirax_interferom": "HIRAX", "tianlai_icyl": "Tianlai",
    "meerkat_b1_interferom": "MeerKAT",
}


def _information(power, noise, weights):
    """Return thermal and CV-limit S/N squared per unit survey volume.

    weights integrates k_perp dk_perp dk_parallel on the positive radial half.
    Thus the prefactor is 1/(4*pi**2), including Gaussian sample variance.
    The stable power fraction avoids squaring very large noise powers.
    """
    fraction = power / (power + noise)
    return (float(np.sum(weights * fraction**2) / (4*np.pi**2)),
            float(np.sum(weights) / (4*np.pi**2)))


def _power_slice(cosmology, instruments, cfg, z_ref, kpar_ref):
    """Compare all instruments at the same redshift and radial wavenumber."""
    pmax = np.sqrt(cfg.kmax**2 - kpar_ref**2)
    p = np.geomspace(min(1e-4, pmax/1000), pmax, 1000)
    parallel = np.full_like(p, kpar_ref)
    geom = cosmology.geometry(z_ref)
    power = cosmology.signal(p, parallel, z_ref)
    rows = []
    for name, instrument in instruments.items():
        _, band = forecast.redshift_bounds(instrument)
        lo = max(z_ref-cfg.delta_z/2, band[0])
        hi = min(z_ref+cfg.delta_z/2, band[1])
        noise = np.full_like(power, np.nan)
        valid = np.zeros_like(p, dtype=bool)
        if band[0] <= z_ref <= band[1] and hi > lo:
            local = dict(instrument, dnutot=instrument["nu_line"]/(1+lo)
                         - instrument["nu_line"]/(1+hi))
            noise, valid = forecast.noise_power(p, parallel, geom, local)
            valid &= np.isfinite(noise) & (noise > 0)
        noise = np.where(valid, noise, np.nan)
        rows.append(pd.DataFrame(dict(
            telescope=name, z=z_ref, k_Mpc_inv=np.hypot(p, parallel),
            k_perp_Mpc_inv=p, k_parallel_Mpc_inv=parallel,
            p_hi_mK2_Mpc3=power, p_noise_mK2_Mpc3=noise,
            power_ratio=power/noise, valid=valid)))
    return pd.concat(rows, ignore_index=True)


def _redshift_snr(cosmology, name, instrument, cfg):
    """Use the same observed z bins and quadrature as full-domain Fisher."""
    requested, band = forecast.redshift_bounds(instrument)
    edges = forecast.bin_edges(*requested, cfg.delta_z)
    rows = []
    cumulative = cumulative_cv = 0.
    for lo, hi in zip(edges[:-1], edges[1:]):
        obslo, obshi = max(lo, band[0]), min(hi, band[1])
        row = dict(telescope=name, z_low=lo, z_high=hi, z_effective=np.nan,
                   status="out_of_band", volume_Mpc3=0.,
                   snr_thermal=np.nan, snr_cv_limit=np.nan,
                   snr_cumulative_thermal=np.nan, snr_cumulative_cv_limit=np.nan,
                   gaussian_mode_count=np.nan)
        if obshi <= obslo:
            rows.append(row)
            continue
        local = dict(instrument, dnutot=instrument["nu_line"]/(1+obslo)
                     - instrument["nu_line"]/(1+obshi))
        zs, wz = forecast.gauss_interval(obslo, obshi, cfg.n_z)
        information = cv_information = volume = weighted_z = 0.
        for z, weight in zip(zs, wz):
            geom = cosmology.geometry(z)
            dv = weight * cosmology.volume_element(z, instrument)
            volume += dv
            weighted_z += dv*z
            grid = forecast.cylindrical_grid(local, geom, cfg)
            if grid is None:
                continue
            p, parallel, weights, valid = grid
            valid &= weights > 0
            if not np.any(valid):
                continue
            power = cosmology.signal(p[valid], parallel[valid], z)
            noise, measured = forecast.noise_power(p[valid], parallel[valid], geom, local)
            if not np.all(measured & np.isfinite(noise) & (noise >= 0)):
                raise FloatingPointError(f"{name}, z={z}: invalid measured-mode noise.")
            thermal, cv = _information(power, noise, weights[valid])
            information += dv*thermal
            cv_information += dv*cv
        cumulative += information
        cumulative_cv += cv_information
        row.update(
            z_effective=weighted_z/volume, volume_Mpc3=volume,
            status="ok" if cv_information > 0 else "no_modes",
            snr_thermal=np.sqrt(information), snr_cv_limit=np.sqrt(cv_information),
            snr_cumulative_thermal=np.sqrt(cumulative),
            snr_cumulative_cv_limit=np.sqrt(cumulative_cv),
            gaussian_mode_count=2*cv_information,
        )
        rows.append(row)
    return pd.DataFrame(rows)


def compute_snr(cfg, surveys=None, *, z_ref=1., kpar_ref=None, reuse_saved=True):
    """Compute diagnostics, or read their matching cache; never run Fisher."""
    surveys = list(forecast.SURVEYS if surveys is None else surveys)
    radial_cap = cfg.kmax
    kpar_ref = min(.005, radial_cap/2) if kpar_ref is None else float(kpar_ref)
    if not surveys or not np.isfinite(z_ref) or z_ref <= 0:
        raise ValueError("Choose at least one survey and a finite positive z_ref.")
    if not 0 < kpar_ref < radial_cap:
        raise ValueError("kpar_ref must be positive and below the total-k ceiling.")
    instruments = {name: forecast.experiment(name, cfg) for name in surveys}
    sources = [Path(forecast.__file__),
               Path(forecast.bf.__file__), Path(forecast.ex.__file__)]
    sources.extend(Path(e["baseline_source"]) for e in instruments.values()
                   if Path(e["baseline_source"]).is_file())
    fingerprint = dict(
        config=asdict(cfg), surveys=surveys, z_ref=z_ref, kpar_ref=kpar_ref,
        calculation_sha256=hashlib.sha256("\n".join(inspect.getsource(function)
            for function in [_information, _power_slice, _redshift_snr, compute_snr]).encode()).hexdigest(),
        sources={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources})
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache_path = out / "snr_diagnostics.joblib"
    if reuse_saved and cache_path.is_file():
        saved = joblib.load(cache_path)
        if saved.get("fingerprint") == fingerprint:
            print("S/N: read matching saved diagnostics; no cosmology or integrals rerun.")
            return saved
    cosmology = forecast.Cosmology(cfg)  # One fiducial CLASS model, no parameter derivatives.
    try:
        curves = _power_slice(cosmology, instruments, cfg, z_ref, kpar_ref)
        bins = []
        for name, instrument in instruments.items():
            bins.append(_redshift_snr(cosmology, name, instrument, cfg))
            print(f"S/N: {name} complete.", flush=True)
    finally:
        cosmology.close()
    redshift = pd.concat(bins, ignore_index=True)
    result = dict(fingerprint=fingerprint, power_slice=curves, redshift=redshift,
                  denominator="(P_HI + P_N)**2", domain="full instrument domain",
                  interpretation="Unmarginalized HI power amplitude, not f_NL significance")
    joblib.dump(result, cache_path)
    redshift.to_csv(out / "snr_by_redshift.csv", index=False)
    return result


def _plot_snr(data, cfg):
    """Draw already computed diagnostics, keeping plotting out of cache keys."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    names = data["fingerprint"]["surveys"]
    curves, redshift = data["power_slice"], data["redshift"]
    z_ref = data["fingerprint"]["z_ref"]
    radial = data["fingerprint"]["kpar_ref"]
    out = Path(cfg.output_dir)
    colors = plt.get_cmap("tab10").colors

    figure_power, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    signal = curves[curves.telescope == names[0]]
    axes[0].loglog(signal.k_Mpc_inv, signal.p_hi_mK2_Mpc3, "k--", lw=2,
                   label=r"$P_{\rm HI}$ (same signal)")
    for i, name in enumerate(names):
        frame = curves[curves.telescope == name]
        axes[0].loglog(frame.k_Mpc_inv, frame.p_noise_mK2_Mpc3,
                       color=colors[i % len(colors)], label=LABELS.get(name, name))
        axes[1].loglog(frame.k_Mpc_inv, frame.power_ratio,
                       color=colors[i % len(colors)], label=LABELS.get(name, name))
    axes[0].set_ylabel(r"Power [mK$^2$ Mpc$^3$]")
    axes[0].set_title("HI signal and instrument thermal noise")
    axes[1].set_ylabel(r"Power ratio $P_{\rm HI}/P_N$")
    axes[1].set_title("Signal-to-thermal-noise power ratio")
    axes[1].axhline(1, color="0.5", lw=1, ls=":")
    for ax in axes:
        ax.set_xlabel(r"Total $k$ [Mpc$^{-1}$]")
        ax.set_xlim(signal.k_Mpc_inv.iloc[0], cfg.kmax)
        ax.grid(True, which="both", alpha=.2)
        ax.legend(fontsize=8)
    figure_power.suptitle(
        rf"$z={z_ref:g}$, $k_\parallel={radial:g}$ Mpc$^{{-1}}$; "
        rf"$k\leq {cfg.kmax:g}$ Mpc$^{{-1}}$; gaps = unavailable modes")
    figure_power.savefig(out / "snr_power_ratio.png", dpi=180)

    figure_snr, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    summary = []
    for i, name in enumerate(names):
        frame = redshift[(redshift.telescope == name) & (redshift.status != "out_of_band")]
        color = colors[i % len(colors)]
        for ax, thermal, cv in [(axes[0], "snr_thermal", "snr_cv_limit"),
                                (axes[1], "snr_cumulative_thermal", "snr_cumulative_cv_limit")]:
            ax.semilogy(frame.z_effective, frame[thermal].where(frame[thermal] > 0),
                         color=color, label=LABELS.get(name, name))
            ax.semilogy(frame.z_effective, frame[cv].where(frame[cv] > 0),
                         color=color, ls="--", alpha=.65)
        last = frame.iloc[-1]
        summary.append(dict(telescope=name, snr_total=last.snr_cumulative_thermal,
                            snr_cv_limit_total=last.snr_cumulative_cv_limit))
    for ax in axes:
        ax.set_xlabel("Redshift z")
        ax.set_ylabel("Power-spectrum detection S/N")
        ax.grid(True, which="both", alpha=.2)
    axes[0].set_title(rf"Per redshift bin ($\Delta z={cfg.delta_z:g}$)")
    axes[1].set_title("Cumulative from each instrument's lowest z")
    axes[0].legend(fontsize=8)
    axes[1].legend(handles=[Line2D([], [], color="0.25", label="Thermal + cosmic variance"),
                           Line2D([], [], color="0.5", ls="--", label="Cosmic-variance limit")],
                   fontsize=8)
    figure_snr.suptitle(
        rf"Full instrument domain: $k_{{\max}}={cfg.kmax:g}$ Mpc$^{{-1}}$, "
        "no independent radial cap; unmarginalized HI amplitude")
    figure_snr.savefig(out / "snr_by_redshift.png", dpi=180)
    return figure_power, figure_snr, pd.DataFrame(summary)


def plot_snr(cfg, surveys=None, *, z_ref=1., kpar_ref=None, reuse_saved=True):
    """Save two figures and return them with a per-survey total-S/N table."""
    import matplotlib

    data = compute_snr(cfg, surveys, z_ref=z_ref, kpar_ref=kpar_ref, reuse_saved=reuse_saved)
    # MathText renders formulas without depending on a local TeX installation.
    with matplotlib.rc_context({
        "text.usetex": False, "font.family": "DejaVu Sans", "font.size": 10,
        "font.weight": "normal", "axes.labelsize": 11, "axes.labelweight": "normal",
        "axes.titlesize": 12, "axes.titleweight": "normal", "figure.titlesize": 12,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
    }):
        return _plot_snr(data, cfg)


if __name__ == "__main__":
    import json
    import matplotlib

    matplotlib.use("Agg")
    config = forecast.Config(**json.loads(
        (forecast.ROOT / "outputs_916/run_config.json").read_text()))
    _, _, totals = plot_snr(config)
    print(totals.to_string(index=False))
