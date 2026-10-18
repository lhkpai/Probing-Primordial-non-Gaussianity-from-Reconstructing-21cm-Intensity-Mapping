"""Check current saved forecasts and notebook settings without historical files."""
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import hashlib
import inspect
import json
import re
import sys

import joblib
import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs_916"
sys.path.insert(0, str(ROOT))
import forecast_916 as forecast


def information_difference(larger, smaller):
    """Return the minimum eigenvalue after removing parameter-unit scales."""
    scale = np.sqrt(np.maximum(np.diag(larger), np.finfo(float).tiny))
    difference = (larger - smaller) / scale[:, None] / scale[None, :]
    minimum = float(np.linalg.eigvalsh((difference + difference.T) / 2).min())
    assert minimum > -1e-8
    return minimum


def check_denominator_comparison(cfg, names, notebook):
    saved = joblib.load(OUT/"fisher_denominator_comparison.joblib")
    expected = dict(
        baseline=forecast._comparison_fingerprint(cfg, names),
        comparison_code=hashlib.sha256(inspect.getsource(
            forecast.compare_fisher_denominators).encode()).hexdigest())
    assert saved["fingerprint"] == expected
    table = pd.read_csv(OUT/"fisher_denominator_comparison.csv")
    pd.testing.assert_frame_equal(table, saved["constraints"], check_dtype=False, rtol=1e-12, atol=1e-14)
    assert len(table) == 2*len(names)*len(forecast.PARAMETERS)
    prior = forecast.planck_fisher()
    traditional = saved["traditional_foreground"]
    assert traditional["fingerprint"] == forecast._traditional_fingerprint(
        saved, cfg, names, traditional["kpar_min_Mpc_inv"])
    assert traditional["added_noise_variance"] is False
    assert traditional["baseline_matrices_reused"] is True
    for model, entries in saved["models"].items():
        assert set(entries) == set(names)
        for name, item in entries.items():
            matrices, constraints = item["matrices"], item["constraints"]
            assert constraints.index.tolist() == forecast.PARAMETERS
            selected = table[(table.covariance_model == model) & (table.telescope == name)].set_index("param")
            pd.testing.assert_frame_equal(selected[constraints.columns], constraints,
                                          check_dtype=False, rtol=1e-12, atol=1e-14)
            assert "analytic_full" not in matrices
            assert not {"sigma_ana_full", "sigma_ana_full_cmb"}.intersection(constraints.columns)
            for label, key in [("sigma_ana_common", "analytic_common"),
                               ("sigma_ana_noise", "analytic_noise"),
                               ("sigma_ana_noise_spline", "analytic_noise_spline"),
                               ("sigma_ana_trad", "analytic_trad")]:
                for suffix, matrix in [("", matrices[key]), ("_cmb", matrices[key]+prior)]:
                    np.testing.assert_allclose(constraints[label+suffix],
                                               forecast.sigma_from_fisher(matrix), rtol=3e-10)
            for key in ["analytic_noise", "analytic_noise_spline"]:
                information_difference(matrices["analytic_common"], matrices[key])
            information_difference(matrices["analytic_common"], matrices["analytic_trad"])
            removed = traditional["removed_matrices"][name][model]
            np.testing.assert_allclose(matrices["analytic_trad"]+removed,
                                       matrices["analytic_common"], rtol=1e-13, atol=1e-10)
            for suffix in ["", "_cmb"]:
                assert (constraints["sigma_ana_trad"+suffix] >=
                        constraints["sigma_ana_common"+suffix]*(1-1e-8)).all()
    for name in names:
        cv = saved["models"]["signal_only"][name]["matrices"]
        thermal = saved["models"]["signal_plus_thermal"][name]["matrices"]
        for key in cv:
            information_difference(cv[key], thermal[key])
    html = [o["data"]["text/html"] for c in notebook.cells if c.cell_type == "code"
            for o in c.outputs if "text/html" in o.get("data", {})]
    for model in saved["models"]:
        fnl = table[(table.covariance_model == model) & (table.param == "f_NL")]
        values = fnl[["sigma_ana_common", "sigma_ana_noise", "sigma_ana_noise_spline"]].to_numpy().ravel()
        assert any(all(f"{value:.4f}" in block for value in values) for block in html)
        values = fnl["sigma_ana_trad"].to_numpy()
        assert any(all(f"{value:.4f}" in block for value in values) for block in html)
    for name in names:
        coverage = traditional["coverage"][name]
        assert (coverage.removed_kpar_max.dropna() < traditional["kpar_min_Mpc_inv"]).all()
    return len(table)


def run():
    cfg = json.loads((OUT / "run_config.json").read_text())
    assert cfg["kmax"] > 0
    assert not {"kpar_max", "voxel_kmax"}.intersection(cfg)
    assert cfg["delta_z"] > 0
    assert cfg["noise_fisher_only"] is False
    assert not {"mock_file", "mock_k_unit", "mock_redshifts"}.intersection(cfg)

    path = ROOT / "9.16-interferometer-telescopes.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    code_cells = [c for c in notebook.cells if c.cell_type == "code"]
    assert all(c.execution_count is not None for c in code_cells)
    assert not [o for c in code_cells for o in c.outputs if o.output_type == "error"]
    # Execute only the settings cell: importing modules and constructing Config
    # does not run CLASS, calculate variances, or overwrite forecast results.
    settings = next(c for c in code_cells if "cfg = Config(" in c.source)
    namespace = {}
    exec(compile(settings.source, str(path), "exec"), namespace)
    assert asdict(namespace["cfg"]) == cfg
    display = "".join(o.get("data", {}).get("text/plain", "") for o in settings.outputs)
    assert repr(namespace["cfg"]) in display

    table = pd.read_csv(OUT / "voxel_noise_all_surveys.csv")
    coverage = pd.read_csv(OUT / "noise_coverage_all_surveys.csv")
    summaries = json.loads((OUT / "noise_calibration_summary.json").read_text())
    comparison_saved = joblib.load(OUT / "noise_interpolation_comparison.joblib")
    assert comparison_saved["fingerprint"] == forecast._comparison_fingerprint(
        namespace["cfg"], namespace["SURVEYS"])
    comparison = pd.read_csv(OUT / "noise_interpolation_comparison.csv").set_index("telescope")
    pd.testing.assert_frame_equal(
        comparison, comparison_saved["comparison"].set_index("telescope"),
        check_dtype=False, rtol=1e-12, atol=1e-14)
    assert table.telescope.unique().tolist() == namespace["SURVEYS"]
    good = table.status != "out_of_band"
    assert np.isfinite(table.loc[good, "var_noise_mK2"]).all()
    assert (table.loc[good, "var_noise_mK2"] > 0).all()
    assert table.loc[~good, "var_noise_mK2"].isna().all()
    np.testing.assert_allclose(table.loc[good, "rms_noise_mK"]**2,
                               table.loc[good, "var_noise_mK2"], rtol=1e-12)

    surveys, individual_tables, individual_coverage = [], [], []
    prior = forecast.planck_fisher()
    for name in namespace["SURVEYS"]:
        saved = joblib.load(OUT / f"fisher_{name}.joblib")
        metadata, matrices = saved["metadata"], saved["matrices"]
        assert metadata["config"] == cfg
        assert saved["parameters"] == forecast.PARAMETERS
        assert metadata["fisher_analytic_denominator"] == forecast.ANALYTIC_DENOMINATOR
        assert metadata["noise_total_denominator"] == forecast.TOTAL_DENOMINATOR
        assert metadata["voxel_variance_reused"] is True
        for source, expected in metadata["voxel_source_sha256"].items():
            assert hashlib.sha256(Path(source).read_bytes()).hexdigest() == expected
        assert metadata["noise_coverage"] == summaries[name]
        assert not any(key.startswith("mock_") for key in metadata)

        voxel = joblib.load(OUT / f"voxel_noise_{name}.joblib")
        # Voxel metadata retains its original provenance; Fisher metadata is new.
        forecast._check_voxel_config(voxel["metadata"]["config"], namespace["cfg"], name)
        if "voxel_source_config" in metadata:
            assert metadata["voxel_source_config"] == voxel["metadata"]["config"]
        individual = pd.read_csv(OUT / f"voxel_noise_{name}.csv")
        pd.testing.assert_frame_equal(individual, voxel["variance_table"],
                                      check_dtype=False, rtol=1e-12, atol=1e-14)
        individual_tables.append(individual)
        selected = pd.read_csv(OUT / f"noise_coverage_{name}.csv")
        individual_coverage.append(selected)
        assert selected.supported_nodes.sum() > 0

        assert "analytic_full" not in matrices
        common, combined = (matrices[k] for k in ["analytic_common", "analytic_noise"])
        noise_min = information_difference(common, combined)
        constraints = pd.read_csv(OUT / f"constraints_{name}.csv", index_col=0)
        assert constraints.index.tolist() == forecast.PARAMETERS
        assert np.isfinite(constraints.to_numpy()).all()
        assert not {"sigma_ana_full", "sigma_ana_full_cmb"}.intersection(constraints.columns)
        for label, matrix in [("sigma_ana_common", common),
                              ("sigma_ana_noise", combined),
                              ("sigma_ana_noise_spline", matrices["analytic_noise_spline"])]:
            np.testing.assert_allclose(constraints[label],
                                       forecast.sigma_from_fisher(matrix), rtol=3e-10)
            np.testing.assert_allclose(constraints[label + "_cmb"],
                                       forecast.sigma_from_fisher(matrix + prior), rtol=3e-10)
        for suffix in ["", "_cmb"]:
            assert (constraints["sigma_ana_noise" + suffix] >=
                    constraints["sigma_ana_common" + suffix] * (1 - 1e-8)).all()
        compared = comparison_saved["surveys"][name]
        assert compared["baseline_max_scaled_difference"] < 1e-8
        for key, value in compared["matrices"].items():
            np.testing.assert_array_equal(value, matrices[key])
        cubic = compared["matrices"]["analytic_noise_spline"]
        information_difference(common, cubic)
        for suffix, matrix in [("", cubic), ("_cmb", cubic+prior)]:
            expected = forecast.sigma_from_fisher(matrix)
            np.testing.assert_allclose(compared["constraints"]["sigma_noise_spline"+suffix],
                                       expected, rtol=3e-10)
            np.testing.assert_allclose(comparison.loc[name, "sigma_noise_spline"+suffix],
                                       expected[-1], rtol=3e-10)
            np.testing.assert_allclose(comparison.loc[name, "sigma_noise_linear"+suffix],
                                       constraints.loc["f_NL", "sigma_ana_noise"+suffix], rtol=3e-10)
        fraction = float(comparison.loc[name, "spline_fallback_mode_fraction"])
        assert 0 <= fraction <= 1
        surveys.append(dict(
            telescope=name, status=summaries[name]["status"],
            supported_nodes=int(selected.supported_nodes.sum()),
            common_minus_noise_min_scaled_eigenvalue=noise_min,
            **{key: float(value) for key, value in constraints.loc["f_NL"].items()}))

    pd.testing.assert_frame_equal(table, pd.concat(individual_tables, ignore_index=True),
                                  check_dtype=False, rtol=1e-12, atol=1e-14)
    pd.testing.assert_frame_equal(coverage, pd.concat(individual_coverage, ignore_index=True),
                                  check_dtype=False, rtol=1e-12, atol=1e-14)
    denominator_rows = check_denominator_comparison(namespace["cfg"], namespace["SURVEYS"], notebook)
    traditional = joblib.load(OUT/"fisher_denominator_comparison.joblib")["traditional_foreground"]
    assert traditional["kpar_min_Mpc_inv"] == namespace["TRAD_KPAR_MIN"]
    log = (OUT / "execution_notebook.log").read_text()
    seconds = float(re.search(r"Notebook execution complete: ([\d.]+)s", log).group(1))
    source_paths = [path, ROOT / "forecast_916.py", ROOT / "voxel_modes_916.py",
                    ROOT / "diagnose_noise_916.py", Path(forecast.bf.__file__),
                    Path(forecast.ex.__file__)]
    summary = dict(
        checked_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        config=cfg, notebook_config_matches_results=True,
        notebook_errors=0, notebook_execution_seconds=seconds,
        total_rows=len(table), valid_rows=int(good.sum()),
        denominator_model=forecast.TOTAL_DENOMINATOR,
        current_result_consistency_passed=True,
        fisher_kmax_Mpc_inv=cfg["kmax"],
        independent_radial_cutoff=False,
        voxel_calibration_kmax_Mpc_inv=float(table.k_max_Mpc_inv.iloc[0]),
        full_domain_parameter_forecast_removed=True,
        voxel_variances_reused=True, fisher_only_update=True,
        spline_comparison_passed=True,
        both_analytic_denominators_checked=True,
        traditional_foreground_comparison_checked=True,
        traditional_kpar_min_Mpc_inv=traditional["kpar_min_Mpc_inv"],
        denominator_comparison_parameter_rows=denominator_rows,
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        surveys=surveys)
    (OUT / "execution_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    run()
