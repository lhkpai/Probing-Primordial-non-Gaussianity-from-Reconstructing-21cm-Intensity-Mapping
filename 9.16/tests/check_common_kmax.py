"""Check current saved forecasts and notebook settings without historical files."""
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import hashlib
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


def run():
    cfg = json.loads((OUT / "run_config.json").read_text())
    assert cfg["kmax"] == cfg["voxel_kmax"] == .363
    assert cfg["delta_z"] == .01
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
        assert voxel["metadata"]["config"] == metadata["config"]
        individual = pd.read_csv(OUT / f"voxel_noise_{name}.csv")
        pd.testing.assert_frame_equal(individual, voxel["variance_table"],
                                      check_dtype=False, rtol=1e-12, atol=1e-14)
        individual_tables.append(individual)
        selected = pd.read_csv(OUT / f"noise_coverage_{name}.csv")
        individual_coverage.append(selected)
        assert selected.supported_nodes.sum() > 0

        full, common, combined = (matrices[k] for k in
                                  ["analytic_full", "analytic_common", "analytic_noise"])
        full_min = information_difference(full, common)
        noise_min = information_difference(common, combined)
        constraints = pd.read_csv(OUT / f"constraints_{name}.csv", index_col=0)
        assert constraints.index.tolist() == forecast.PARAMETERS
        assert np.isfinite(constraints.to_numpy()).all()
        for label, matrix in [("sigma_ana_full", full),
                              ("sigma_ana_common", common),
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
            full_minus_common_min_scaled_eigenvalue=full_min,
            common_minus_noise_min_scaled_eigenvalue=noise_min,
            **{key: float(value) for key, value in constraints.loc["f_NL"].items()}))

    pd.testing.assert_frame_equal(table, pd.concat(individual_tables, ignore_index=True),
                                  check_dtype=False, rtol=1e-12, atol=1e-14)
    pd.testing.assert_frame_equal(coverage, pd.concat(individual_coverage, ignore_index=True),
                                  check_dtype=False, rtol=1e-12, atol=1e-14)
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
        voxel_variances_reused=True, fisher_only_update=True,
        spline_comparison_passed=True,
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        surveys=surveys)
    (OUT / "execution_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    run()
