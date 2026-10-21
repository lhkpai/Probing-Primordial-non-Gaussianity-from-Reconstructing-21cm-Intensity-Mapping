"""Independent domain and numerical checks for the radial-cut comparison."""
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import compare_foreground_cuts_916 as scan
import forecast_916 as forecast


def scaled_difference(reference, estimate):
    scale = np.sqrt(np.diag(reference))
    return float(np.max(np.abs((estimate-reference)/scale[:, None]/scale[None, :])))


def check():
    cfg = forecast.Config(**json.loads((ROOT/"outputs_916/run_config.json").read_text()))
    models = noise = None
    try:
        models = forecast.SignalModels(cfg)
        noise = forecast.NoiseCalibration(cfg)
        name, z = "hirax_interferom", 2.501
        table = forecast.pd.read_csv(ROOT/f"outputs_916/voxel_noise_{name}.csv")
        valid = np.isfinite(table.var_noise_mK2) & (table.var_noise_mK2 > 0)
        sigma = float(np.sqrt(np.interp(z, table.loc[valid, "z_mid"],
                                       table.loc[valid, "var_noise_mK2"])))
        instrument = dict(forecast.experiment(name, cfg), dnutot=100.)
        cuts = scan.DEFAULT_CUTS
        actual, _ = scan.fisher_at_cuts(models, instrument, z, cfg, noise, sigma, cuts)
        geometry = models.fid.geometry(z)
        errors = {}
        for index, cut in enumerate(cuts):
            # Independent direct integration over the complete retained domain,
            # rather than summing the disjoint bands used by the scan.
            domain = SimpleNamespace(p=noise.p, ell=np.r_[cut, noise.ell[noise.ell > cut]])
            p, ell, weights, selected = forecast.cylindrical_grid(instrument, geometry, cfg, domain)
            query = noise.evaluate_amplitude(p, ell, z, sigma)
            selected &= (weights > 0) & query["valid"]
            assert ell[selected].min() >= cut
            assert np.hypot(p[selected], ell[selected]).max() <= cfg.kmax*(1+1e-12)
            power, derivatives = models.power_and_derivatives(p[selected], ell[selected], z)
            thermal, _ = forecast.noise_power(p[selected], ell[selected], geometry, instrument)
            w = weights[selected]/(4*np.pi**2)
            denominators = {"signal_only": power**2,
                            "signal_plus_thermal": (power+thermal)**2}
            for model, denominator in denominators.items():
                independent = (derivatives*(w/denominator)) @ derivatives.T
                error = scaled_difference(independent, actual[model][index])
                assert error < 1e-4, (cut, model, error)
                errors[f"{cut:g}/{model}"] = error
        refined, _ = scan.fisher_at_cuts(models, instrument, z,
            replace(cfg, n_perp=192, n_parallel=128), noise, sigma, cuts)
        refinement, sigma_refinement = {}, {}
        for model in scan.MODELS:
            for index, cut in enumerate(cuts):
                error = scaled_difference(refined[model][index], actual[model][index])
                assert error < 1e-3, (cut, model, error)
                refinement[f"{cut:g}/{model}"] = error
                coarse_sigma = forecast.sigma_from_fisher(actual[model][index])
                fine_sigma = forecast.sigma_from_fisher(refined[model][index])
                if np.all(np.isfinite(coarse_sigma) & np.isfinite(fine_sigma)):
                    change = float(np.max(np.abs(fine_sigma/coarse_sigma-1)))
                    assert change < 1e-2, (cut, model, change)
                    sigma_refinement[f"{cut:g}/{model}"] = change
                if index:
                    difference = actual[model][index-1]-actual[model][index]
                    scale = np.sqrt(np.diag(actual[model][index-1]))
                    assert np.linalg.eigvalsh(difference/scale[:, None]/scale[None, :]).min() > -1e-10
        reference = noise.evaluate_amplitude

        def changed_amplitude(*args, **kwargs):
            values = reference(*args, **kwargs)
            values["A_noise_conditional"] *= 100
            return values

        with patch.object(noise, "evaluate_amplitude", side_effect=changed_amplitude):
            changed, _ = scan.fisher_at_cuts(models, instrument, z, cfg, noise, sigma, cuts)
        for model in scan.MODELS:
            np.testing.assert_array_equal(changed[model], actual[model])
    finally:
        if models is not None:
            models.close()
        if noise is not None:
            noise.close()
    report = dict(telescope=name, z=z, cuts_Mpc_inv=list(cuts),
                  independent_direct_integral_max_scaled_difference=errors,
                  refinement_max_scaled_difference=refinement,
                  refinement_max_relative_marginalized_sigma_change=sigma_refinement,
                  nested_domain_information_checked=True,
                  added_noise_variance=False, voxel_variances_recalculated=False)
    path = ROOT/"outputs_916/foreground_cutoff_validation.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report, indent=2))


def check_saved():
    """Verify every production matrix/error and cache reuse without integrating."""
    out = ROOT/"outputs_916"
    cfg = forecast.Config(**json.loads((out/"run_config.json").read_text()))
    saved = joblib.load(out/"foreground_cutoff_comparison.joblib")
    baseline = joblib.load(out/"fisher_denominator_comparison.joblib")
    cuts = np.asarray(scan.DEFAULT_CUTS)
    assert saved["fingerprint"] == scan._fingerprint(cfg, forecast.SURVEYS, cuts)
    assert set(saved["surveys"]) == set(forecast.SURVEYS)
    table = pd.read_csv(out/"foreground_cutoff_comparison.csv")
    pd.testing.assert_frame_equal(table, saved["constraints"], rtol=1e-12, atol=1e-14)
    assert len(table) == 210
    assert np.isfinite(table[["sigma", "sigma_cmb"]].to_numpy()).all()
    prior = forecast.planck_fisher()
    minimum = np.inf
    for name, item in saved["surveys"].items():
        coverage = item["coverage"].pivot(index="z", columns="kpar_min_Mpc_inv", values="mode_weight")
        assert (np.diff(coverage.to_numpy(), axis=1) <= 1e-12).all()
        for model in scan.MODELS:
            previous = baseline["models"][model][name]["matrices"]["analytic_trad"]
            for index, cut in enumerate(cuts):
                matrix = item["matrices"][model][index]
                np.testing.assert_allclose(matrix, matrix.T, rtol=1e-13, atol=1e-12)
                scale = np.sqrt(np.diag(previous))
                eigenvalue = float(np.linalg.eigvalsh(
                    (previous-matrix)/scale[:, None]/scale[None, :]).min())
                assert eigenvalue > -1e-8, (name, model, cut, eigenvalue)
                minimum = min(minimum, eigenvalue)
                selected = item["constraints"].loc[
                    (item["constraints"].covariance_model == model)
                    & (item["constraints"].kpar_min_Mpc_inv == cut)].set_index("param")
                np.testing.assert_allclose(selected.loc[forecast.PARAMETERS, "sigma"],
                                           forecast.sigma_from_fisher(matrix), rtol=1e-10)
                np.testing.assert_allclose(selected.loc[forecast.PARAMETERS, "sigma_cmb"],
                                           forecast.sigma_from_fisher(matrix+prior), rtol=1e-10)
                assert (forecast.sigma_from_fisher(matrix)
                        >= forecast.sigma_from_fisher(previous)*(1-1e-8)).all()
                previous = matrix
        for cv, thermal in zip(item["matrices"]["signal_only"],
                               item["matrices"]["signal_plus_thermal"]):
            scale = np.sqrt(np.diag(cv))
            assert np.linalg.eigvalsh((cv-thermal)/scale[:, None]/scale[None, :]).min() > -1e-8
    forbidden = AssertionError("Matching cut comparison must not recalculate cosmologies or Noise.")
    with patch.object(forecast.SignalModels, "__init__", side_effect=forbidden), \
         patch.object(forecast.NoiseCalibration, "__init__", side_effect=forbidden):
        loaded = scan.compare_foreground_cuts(cfg)
    pd.testing.assert_frame_equal(loaded["constraints"], saved["constraints"], check_exact=True)
    report_path = out/"foreground_cutoff_validation.json"
    report = json.loads(report_path.read_text())
    report["production"] = dict(
        surveys=len(saved["surveys"]), models=len(scan.MODELS), cuts_Mpc_inv=cuts.tolist(),
        parameter_rows=len(table), matrices_checked=30,
        minimum_scaled_information_difference_eigenvalue=minimum,
        covariance_and_cmb_verified=True, errors_monotonic_with_cut=True,
        matching_cache_reused_without_calculation=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report["production"], indent=2))


if __name__ == "__main__":
    if "--saved-only" in sys.argv:
        check_saved()
    else:
        check()
