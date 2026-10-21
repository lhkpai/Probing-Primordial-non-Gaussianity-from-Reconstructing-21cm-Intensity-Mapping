"""Compare analytic Fisher forecasts at several added k_parallel lower bounds.

Reuse the baseline redshift nodes, voxel calibration and common-domain mask.
Disjoint radial bands share one signal/derivative evaluation at each redshift;
their cumulative sums give nested cuts without subtracting nearly equal matrices.
Wavenumbers are comoving Mpc^-1. No reconstructed Noise variance is added.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd

import forecast_916 as forecast

MODELS = ("signal_only", "signal_plus_thermal")
DEFAULT_CUTS = (.05, .1, .2)


def _band_grid(instrument, geometry, cfg, noise, lower, upper):
    """Put each radial boundary and spherical intersection on a segment edge."""
    lower, upper = max(lower, noise.ell[0]), min(upper, noise.ell[-1])
    if upper <= lower:
        return None
    _, cap = forecast.bf.noise_k_limits(geometry, instrument)
    knee = np.sqrt(max(0., cap**2-upper**2))
    transverse = noise.p
    if transverse[0] < knee < transverse[-1]:
        transverse = np.unique(np.r_[transverse, knee])
    parallel = np.r_[lower, noise.ell[(noise.ell > lower) & (noise.ell < upper)], upper]
    return forecast.cylindrical_grid(
        instrument, geometry, cfg, SimpleNamespace(p=transverse, ell=parallel))


def fisher_at_cuts(models, instrument, z, cfg, noise, sigma_mK, cuts):
    """Integrate retained bands directly; evaluate both covariances together."""
    cuts = np.asarray(cuts, dtype=float)
    shape = (len(cuts), len(forecast.PARAMETERS), len(forecast.PARAMETERS))
    bands = {model: np.zeros(shape) for model in MODELS}
    mode_weights = np.zeros(len(cuts))
    geometry = models.fid.geometry(z)
    pieces = []
    for index, (lower, upper) in enumerate(zip(cuts, np.r_[cuts[1:], cfg.kmax])):
        grid = _band_grid(instrument, geometry, cfg, noise, lower, upper)
        if grid is None:
            continue
        p, ell, weights, valid = grid
        valid &= weights > 0
        if valid.any():
            pieces.append((p[valid], ell[valid], weights[valid],
                           np.full(int(valid.sum()), index, dtype=np.int8)))
    if not pieces:
        return bands, mode_weights
    p, ell, weights, band_index = (np.concatenate(parts) for parts in zip(*pieces))
    query = noise.evaluate_amplitude(p, ell, z, sigma_mK)
    amplitude = query["A_noise_conditional"]
    valid = query["valid"] & np.isfinite(amplitude) & (amplitude >= 0)
    if not valid.any():
        return bands, mode_weights
    p, ell, weights, band_index = (values[valid] for values in (p, ell, weights, band_index))
    power, derivatives = models.power_and_derivatives(p, ell, z)
    thermal, measured = forecast.noise_power(p, ell, geometry, instrument)
    if not np.all(measured & np.isfinite(power) & (power > 0)
                  & np.isfinite(thermal) & (thermal > 0)):
        raise FloatingPointError("Invalid power inside the retained common domain.")
    logarithmic = derivatives/power
    w = weights/(4*np.pi**2)
    thermal_weight = w/(1+thermal/power)**2
    for index in range(len(cuts)):
        select = band_index == index
        response = logarithmic[:, select]
        bands["signal_only"][index] = (response*w[select]) @ response.T
        bands["signal_plus_thermal"][index] = (response*thermal_weight[select]) @ response.T
        mode_weights[index] = weights[select].sum()
    retained = {model: np.cumsum(values[::-1], axis=0)[::-1] for model, values in bands.items()}
    return retained, np.cumsum(mode_weights[::-1])[::-1]


def _survey_forecast(models, instrument, frame, coverage, cfg, noise, cuts):
    """Use exactly the saved common-domain z, RMS and volume quadrature."""
    shape = (len(cuts), len(forecast.PARAMETERS), len(forecast.PARAMETERS))
    matrices = {model: np.zeros(shape) for model in MODELS}
    sample_z = frame.loc[np.isfinite(frame.var_noise_mK2) & (frame.var_noise_mK2 > 0),
                         "z_mid"].sort_values().to_numpy()
    if not len(sample_z):
        raise ValueError("No saved voxel RMS samples for the common-domain forecast.")
    rows = list(frame.itertuples(index=False))
    records = []
    nodes = coverage.loc[coverage.status == "ok"]
    for count, node in enumerate(nodes.itertuples(index=False), start=1):
        row = rows[int(np.searchsorted(frame.z_high.to_numpy(), node.z, side="right"))]
        lo = max(row.z_obs_low, noise.z[0], sample_z[0])
        hi = min(row.z_obs_high, noise.z[-1], sample_z[-1])
        local = dict(instrument, dnutot=instrument["nu_line"]/(1+lo)
                     - instrument["nu_line"]/(1+hi))
        values, weights = fisher_at_cuts(models, local, node.z, cfg, noise, node.sigma_mK, cuts)
        for model in MODELS:
            matrices[model] += node.volume_weight_Mpc3*values[model]
        for cut, weight in zip(cuts, weights):
            records.append(dict(z=node.z, sigma_mK=node.sigma_mK,
                                kpar_min_Mpc_inv=cut, mode_weight=float(weight),
                                volume_weight_Mpc3=node.volume_weight_Mpc3))
        if count % 50 == 0 or count == len(nodes):
            print(f"[{row.telescope}] radial cuts: z node {count}/{len(nodes)}", flush=True)
    return matrices, pd.DataFrame(records)


def _fingerprint(cfg, surveys, cuts):
    out = Path(cfg.output_dir)
    sources = [Path(__file__), Path(forecast.__file__), Path(forecast.bf.__file__),
               Path(forecast.ex.__file__), forecast.NOISE_ROOT/"noise_hkl.py"]
    inputs = [out/"fisher_denominator_comparison.joblib"]
    inputs += [out/f"{kind}_{name}.{extension}" for name in surveys
               for kind, extension in [("voxel_noise", "joblib"), ("noise_coverage", "csv")]]
    stat = Path(cfg.noise_file).stat()
    return dict(schema=1, config=asdict(cfg), surveys=surveys, cuts_Mpc_inv=list(cuts),
                sources_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                inputs_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                noise_file_size=stat.st_size, noise_file_mtime_ns=stat.st_mtime_ns)


def _save(saved, out):
    tables = [item["constraints"] for item in saved["surveys"].values()]
    saved["constraints"] = pd.concat(tables, ignore_index=True)
    joblib.dump(saved, out/"foreground_cutoff_comparison.joblib")
    saved["constraints"].to_csv(out/"foreground_cutoff_comparison.csv", index=False)


def compare_foreground_cuts(cfg=None, surveys=None, cuts=DEFAULT_CUTS, *, reuse_saved=True):
    """Compute only missing surveys, or reuse matching comparison checkpoints."""
    cfg = cfg or forecast.Config(**json.loads(
        (forecast.ROOT/"outputs_916/run_config.json").read_text()))
    out = Path(cfg.output_dir)
    surveys = list(forecast.SURVEYS if surveys is None else surveys)
    cuts = np.asarray(cuts, dtype=float)
    if (cuts.ndim != 1 or not len(cuts) or not np.all(np.isfinite(cuts))
            or np.any(cuts <= 0) or np.any(cuts >= cfg.kmax) or np.any(np.diff(cuts) <= 0)):
        raise ValueError("Cuts must be strictly increasing and lie between zero and kmax.")
    baseline = joblib.load(out/"fisher_denominator_comparison.joblib")
    expected = dict(baseline=forecast._comparison_fingerprint(cfg, surveys),
                    comparison_code=hashlib.sha256(inspect.getsource(
                        forecast.compare_fisher_denominators).encode()).hexdigest())
    if baseline["fingerprint"] != expected:
        raise ValueError("The comparison requires a matching saved Fisher baseline.")
    fingerprint = _fingerprint(cfg, surveys, cuts)
    path = out/"foreground_cutoff_comparison.joblib"
    saved = joblib.load(path) if reuse_saved and path.is_file() else {}
    if saved.get("fingerprint") != fingerprint:
        saved = dict(fingerprint=fingerprint, parameters=forecast.PARAMETERS, surveys={},
                     method="direct disjoint radial bands; cumulative retained information",
                     added_noise_variance=False, voxel_variances_reused=True)
    pending = [name for name in surveys if name not in saved["surveys"]]
    if not pending:
        print("Using saved radial-cut comparison; no integrals recalculated.", flush=True)
        _save(saved, out)
        return saved
    prior = forecast.planck_fisher()
    models = noise = None
    try:
        models = forecast.SignalModels(cfg)
        noise = forecast.NoiseCalibration(cfg)
        for name in pending:
            voxel = joblib.load(out/f"voxel_noise_{name}.joblib")
            forecast._check_voxel_config(voxel["metadata"]["config"], cfg, name)
            coverage = pd.read_csv(out/f"noise_coverage_{name}.csv", float_precision="round_trip")
            matrices, records = _survey_forecast(
                models, forecast.experiment(name, cfg), voxel["variance_table"], coverage,
                cfg, noise, cuts)
            tables = []
            for model in MODELS:
                for index, cut in enumerate(cuts):
                    matrix = (matrices[model][index]+matrices[model][index].T)/2
                    matrices[model][index] = matrix
                    sigma = forecast.sigma_from_fisher(matrix)
                    cmb = (forecast.sigma_from_fisher(matrix+prior) if np.any(np.diag(matrix) > 0)
                           else np.full(len(forecast.PARAMETERS), np.nan))
                    tables.append(pd.DataFrame(dict(
                        telescope=name, covariance_model=model, kpar_min_Mpc_inv=cut,
                        param=forecast.PARAMETERS, sigma=sigma, sigma_cmb=cmb)))
            saved["surveys"][name] = dict(matrices=matrices, coverage=records,
                                         constraints=pd.concat(tables, ignore_index=True))
            _save(saved, out)
            print(f"[{name}] saved all {len(cuts)} radial cuts.", flush=True)
    finally:
        if models is not None:
            models.close()
        if noise is not None:
            noise.close()
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuts", nargs="+", type=float, default=list(DEFAULT_CUTS))
    parser.add_argument("--force", action="store_true", help="Recalculate this comparison only.")
    args = parser.parse_args()
    result = compare_foreground_cuts(cuts=args.cuts, reuse_saved=not args.force)
    print(result["constraints"].query("param == 'f_NL'").to_string(index=False))


if __name__ == "__main__":
    main()
