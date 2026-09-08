#!/usr/bin/env python3
"""Print all input parameters for a chosen SKA-MID experiment config.

This is a lightweight inspector for bao21cm-master experiments:
- reads experiment dicts from radiofisher.experiments (e.g. SKA1MIDbase1)
- prints cosmology + survey + instrument parameters in categories
- optionally prints the *effective* experiment parameters in a redshift bin
  using radiofisher.baofisher.overlapping_expts (handles overlap configs)

Examples:
  python3 print_ska_mid_inputs.py --expt SKA1MIDbase1
  python3 print_ska_mid_inputs.py --expt SKA1MIDbase2 --zmin 0.9 --zmax 1.1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _pick(d: Dict[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in keys:
        if k in d:
            out[k] = d[k]
    return out


_MEANING: Dict[str, str] = {
    # Cosmology
    "omega_M_0": "Present-day total matter density fraction Ωm",
    "omega_lambda_0": "Present-day dark energy density fraction ΩΛ",
    "omega_b_0": "Present-day baryon density fraction Ωb",
    "omega_HI_0": "Present-day HI density fraction ΩHI",
    "h": "Dimensionless Hubble parameter (H0 = 100 h km/s/Mpc)",
    "N_eff": "Effective number of relativistic species",
    "mnu": "Sum of neutrino masses [eV] (if used)",
    "ns": "Scalar spectral index (tilt of primordial spectrum)",
    "sigma_8": "RMS linear matter fluctuation at 8 h⁻¹ Mpc",
    "gamma": "Growth index in f≈Ωm(z)^γ",
    "w0": "Dark energy equation of state at a=1",
    "wa": "Dark energy evolution parameter (CPL)",
    "k_piv": "Pivot scale for ns [1/Mpc]",
    "aperp": "Transverse AP dilation parameter α⊥ (fiducial=1)",
    "apar": "Line-of-sight AP dilation parameter α∥ (fiducial=1)",
    "bHI0": "HI bias normalization at z~0 used by the model",
    "sigma_nl": "Nonlinear BAO damping scale (smooths BAO wiggles)",
    "fNL": "Local-type primordial non-Gaussianity parameter",
    # Survey/global
    "ttot": "Total integration time (converted to internal units)",
    "nu_line": "Rest-frame 21cm line frequency [MHz]",
    "epsilon_fg": "Foreground subtraction residual amplitude",
    "k_nl0": "Nonlinear scale at z=0 used to set kmax",
    # Instrument/experiment
    "mode": "Observation mode: dish/interferom/combined",
    "Ndish": "Number of dishes",
    "Nbeam": "Number of beams / feeds",
    "Ddish": "Dish diameter [m]",
    "Tinst": "Instrument/system temperature [mK]",
    "survey_dnutot": "Total survey bandwidth [MHz]",
    "survey_numax": "Maximum observing frequency [MHz]",
    "dnu": "Channel width [MHz]",
    "Sarea": "Survey area [sr]",
    "Dmin": "Minimum interferometer baseline [m]",
    "Dmax": "Maximum interferometer baseline [m]",
    "nu_crit": "Critical frequency for PAF performance [MHz]",
    "n(x)": "Baseline/antenna density file for interferometer sensitivity",
    "use": "Flags for which Fisher contributions are enabled",
    "overlap": "Overlapping-instrument config (effective params computed per bin)",
    # Fisher run-time
    "kmin": "Minimum k used in Fisher integral [1/Mpc]",
    "kmax": "Maximum k used in Fisher integral [1/Mpc]",
    "nsamp_k": "Number of k samples (log grid) for Fisher integral",
    "nsamp_u": "Number of μ samples for Fisher integral",
    "dz": "Redshift bin width if zmin/zmax not provided",
    "cache_pk": "Matter power spectrum cache file (k, P(k))",
    "pk_npoints": "Number of k points when generating cache_pk via python camb",
    "pk_kmax": "Max k when generating cache_pk via python camb [1/Mpc]",
}

# Where each parameter is used in the code (high-level index).
_WHERE: Dict[str, str] = {
    # Binning / survey band
    "survey_numax": "baofisher.zbins_fixed/zbins_equal_spaced/zbins_const_dr (band edges)",
    "survey_dnutot": "baofisher.zbins_fixed/zbins_equal_spaced/zbins_const_dr (band edges)",
    "nu_line": "baofisher.zbins_* and noise/beam conversions (ν(z))",
    "dnu": "baofisher.noise_rms_per_voxel* (thermal noise per voxel)",
    # Survey strategy
    "Sarea": "baofisher.noise_rms_per_voxel* and survey volume/mode-counting",
    "ttot": "baofisher.noise_rms_per_voxel* (thermal noise)",
    # Instrument
    "Tinst": "baofisher.noise_rms_per_voxel* (Tsys = Tinst + Tsky)",
    "Ndish": "baofisher.noise_rms_per_voxel and interferometer_response (multiplicity)",
    "Nbeam": "baofisher.noise_rms_per_voxel (multiplicity)",
    "Ddish": "baofisher.noise_rms_per_voxel (beam) and interferometer_response (FoV)",
    "n(x)": "baofisher.interferometer_response (baseline density n(u))",
    "Dmin": "baofisher.interferometer_response (uniform n(u) case)",
    "Dmax": "baofisher.interferometer_response (uniform n(u) case)",
    # Foregrounds/nonlinear
    "epsilon_fg": "baofisher.foreground_residual (adds residual covariance)",
    "k_nl0": "baofisher.fisher (sets kmax via nonlinear scale)",
    "sigma_nl": "baofisher.spline_pk_nobao / BAO damping model",
    # Cosmology background/growth
    "omega_M_0": "baofisher.background_evolution_splines (E(z), distances)",
    "omega_lambda_0": "baofisher.background_evolution_splines (E(z), distances)",
    "omega_b_0": "power spectrum generation (CAMB) and transfer function",
    "h": "baofisher.background_evolution_splines and P(k) units",
    "w0": "baofisher.background_evolution_splines (E(z))",
    "wa": "baofisher.background_evolution_splines (E(z))",
    "gamma": "baofisher.background_evolution_splines (growth rate f≈Ωm^γ)",
    "ns": "power spectrum shape (CAMB) and spline_pk_nobao",
    "sigma_8": "power spectrum normalization (CAMB As rescale)",
    "omega_HI_0": "baofisher.Tb / IM signal amplitude",
    "bHI0": "baofisher.bias_HI / IM signal amplitude",
    "foregrounds": "baofisher foreground/noise model",
    "use": "baofisher.fisher (toggles which constraints enter)",
    # AP parameters are used inside the Fisher derivatives
    "aperp": "baofisher.fisher (AP mapping: k⊥ and distances)",
    "apar": "baofisher.fisher (AP mapping: k∥ and rν)",
    # Runtime
    "kmin": "run_fisher_sigmas.py -> baofisher.fisher(kmin,kmax)",
    "kmax": "run_fisher_sigmas.py -> baofisher.fisher(kmin,kmax)",
    "nsamp_k": "run_fisher_sigmas.py sets baofisher.NSAMP_K",
    "nsamp_u": "run_fisher_sigmas.py sets baofisher.NSAMP_U",
    "cache_pk": "run_fisher_sigmas.py / print_ska_mid_inputs.py (P(k) input)",
    "pk_npoints": "run_fisher_sigmas.py (CAMB cache generation)",
    "pk_kmax": "run_fisher_sigmas.py (CAMB cache generation)",
}


def _fmt_item(k: str, v: Any) -> str:
    base = k.split(".", 1)[0]
    meaning = _MEANING.get(k) or _MEANING.get(base)
    where = _WHERE.get(k) or _WHERE.get(base)
    parts: List[str] = [str(v)]
    if meaning:
        parts.append(f"# {meaning}")
    if where:
        parts.append(f"| used: {where}")
    return "  ".join(parts)


def _print_section(title: str, items: List[Tuple[str, Any]]) -> None:
    print(f"\n[{title}]")
    if not items:
        print("  (none)")
        return
    width = max(len(k) for k, _ in items)
    for k, v in items:
        print(f"  {k:<{width}s} : {_fmt_item(k, v)}")


def _sorted_items(d: Dict[str, Any]) -> List[Tuple[str, Any]]:
    return [(k, d[k]) for k in sorted(d.keys())]


def _flatten(d: Any, prefix: str = "") -> List[Tuple[str, Any]]:
    """Flatten nested dicts into dot-separated keys for full-value printing."""
    items: List[Tuple[str, Any]] = []
    if isinstance(d, dict):
        for k in sorted(d.keys()):
            key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
            items.extend(_flatten(d[k], key))
        return items
    return [(prefix, d)]


def _print_indexed(
    title: str, items: List[Tuple[str, Any]], start_index: int = 1
) -> int:
    """Print a numbered index list and return the next index."""
    print(f"\n[{title}]")
    if not items:
        print("  (none)")
        return start_index
    idx = start_index
    width = max(len(k) for k, _ in items)
    for k, v in items:
        print(f"  [{idx:03d}] {k:<{width}s} = {_fmt_item(k, v)}")
        idx += 1
    return idx


def split_cosmo(cosmo: Dict[str, Any]) -> List[Tuple[str, List[Tuple[str, Any]]]]:
    # Group keys by physics/theme.
    groups: List[Tuple[str, List[str]]] = [
        (
            "Background",
            ["omega_M_0", "omega_lambda_0", "omega_b_0", "h", "N_eff", "mnu"],
        ),
        ("DarkEnergy", ["w0", "wa"]),
        ("Primordial", ["ns", "k_piv"]),
        ("Growth", ["sigma_8", "gamma", "gamma0", "gamma1"]),
        ("HI", ["omega_HI_0", "bHI0"]),
        ("AP/Geometry", ["aperp", "apar"]),
        ("BAO/Nonlinear", ["sigma_nl"]),
        ("ModifiedGravity", ["eta0", "eta1", "A_xi", "logkmg"]),
        ("BiasModel", ["b_1", "k0_bias"]),
        ("Amplitude/Nuisance", ["A", "fNL"]),
    ]

    used = set()
    out: List[Tuple[str, List[Tuple[str, Any]]]] = []
    for label, keys in groups:
        picked = _pick(cosmo, keys)
        used |= set(picked.keys())
        out.append((label, _sorted_items(picked)))

    # Foregrounds is a nested dict
    fg = cosmo.get("foregrounds")
    if isinstance(fg, dict):
        used.add("foregrounds")
        out.append(("Foregrounds", _sorted_items(fg)))

    # Anything left
    remaining = {k: v for k, v in cosmo.items() if k not in used}
    if remaining:
        out.append(("Other", _sorted_items(remaining)))
    return out


def split_expt(expt: Dict[str, Any]) -> List[Tuple[str, List[Tuple[str, Any]]]]:
    groups: List[Tuple[str, List[str]]] = [
        ("Mode", ["mode"]),
        ("Array/Beams", ["Ndish", "Nbeam", "Ddish", "Dmin", "Dmax", "nu_crit"]),
        ("Receiver", ["Tinst"]),
        ("Frequency", ["survey_dnutot", "survey_numax", "dnu", "nu_line"]),
        ("Survey", ["Sarea", "ttot"]),
        ("Foreground/Nonlinear", ["epsilon_fg", "k_nl0"]),
        ("InterferometerDensity", ["n(x)"]),
        ("Overlap", ["overlap"]),
        ("UseFlags", ["use"]),
    ]

    used = set()
    out: List[Tuple[str, List[Tuple[str, Any]]]] = []
    for label, keys in groups:
        picked = _pick(expt, keys)
        used |= set(picked.keys())
        out.append((label, _sorted_items(picked)))

    remaining = {k: v for k, v in expt.items() if k not in used}
    if remaining:
        out.append(("Other", _sorted_items(remaining)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--expt",
        nargs="+",
        default=["SKA1MIDbase1"],
        help="One or more experiment names from radiofisher.experiments (e.g. SKA1MIDbase1 SKA1MIDbase2)",
    )
    ap.add_argument("--zmin", type=float, default=None)
    ap.add_argument("--zmax", type=float, default=None)
    ap.add_argument(
        "--show-nx-head",
        type=int,
        default=0,
        help="If >0, show first N lines of the n(x) file",
    )
    ap.add_argument(
        "--cache-pk",
        default=None,
        help="Optional: path to cache_pk.dat to record as an input",
    )
    # Fisher runtime knobs (match run_fisher_sigmas.py flags)
    ap.add_argument("--kmin", type=float, default=None)
    ap.add_argument("--kmax", type=float, default=None)
    ap.add_argument("--nsamp-k", type=int, default=None)
    ap.add_argument("--nsamp-u", type=int, default=None)
    ap.add_argument("--dz", type=float, default=None)
    ap.add_argument("--pk-npoints", type=int, default=None)
    ap.add_argument("--pk-kmax", type=float, default=None)
    ap.add_argument(
        "--index",
        action="store_true",
        help="Also print a full indexed list of all input keys and values",
    )
    args = ap.parse_args()

    import radiofisher.experiments as experiments

    cosmo = experiments.cosmo
    print("=== bao21cm-master input parameter dump ===")
    print(f"Experiments: {' '.join(args.expt)}")

    # Fisher runtime settings (if provided)
    run_settings: Dict[str, Any] = {}
    for k in ["kmin", "kmax", "nsamp_k", "nsamp_u", "dz", "pk_npoints", "pk_kmax"]:
        # argparse uses nsamp_k, nsamp_u attribute names
        attr = k
        if k == "nsamp_k":
            attr = "nsamp_k"
        if k == "nsamp_u":
            attr = "nsamp_u"
        val = getattr(args, attr, None)
        if val is not None:
            run_settings[k] = val
    if args.cache_pk:
        run_settings["cache_pk"] = args.cache_pk
    if run_settings:
        print("\n=== Fisher run-time settings (what you pass on the command line) ===")
        _print_section("RunSettings", _sorted_items(run_settings))

        if args.index:
            _print_indexed(
                "RunSettings (indexed)", _flatten(run_settings), start_index=1
            )

    # Cosmology (shared)
    print("\n=== Cosmology (experiments.cosmo) ===")
    for label, items in split_cosmo(cosmo):
        _print_section(label, items)

    if args.index:
        _print_indexed("Cosmology (indexed)", _flatten(cosmo), start_index=1)

    # Loop experiments
    for expt_name in args.expt:
        if not hasattr(experiments, expt_name):
            raise SystemExit(
                f"Unknown --expt {expt_name!r}. Try e.g. SKA1MIDbase1 SKA1MIDbase2"
            )

        expt = getattr(experiments, expt_name)
        print(f"\n\n=== Experiment: {expt_name} (from experiments.py) ===")
        for label, items in split_expt(expt):
            _print_section(label, items)

        if args.index:
            _print_indexed("Experiment inputs (indexed)", _flatten(expt), start_index=1)

        # Data files
        nxf = expt.get("n(x)")
        if isinstance(nxf, str):
            nx_path = Path(nxf)
            if not nx_path.is_absolute():
                nx_path = Path(__file__).resolve().parent / nx_path
            print("\n=== Data files ===")
            print(f"n(x) file: {nx_path}  # {_MEANING.get('n(x)', '')}")
            print(f"exists : {nx_path.exists()}")
            if args.show_nx_head and nx_path.exists():
                print(f"\n--- head -n {args.show_nx_head} {nx_path} ---")
                with nx_path.open("r", encoding="utf-8", errors="replace") as f:
                    for _ in range(args.show_nx_head):
                        line = f.readline()
                        if not line:
                            break
                        print(line.rstrip("\n"))

        if args.cache_pk:
            cache_pk = Path(args.cache_pk)
            if not cache_pk.is_absolute():
                cache_pk = Path(__file__).resolve().parent / cache_pk
            print(
                f"\ncache_pk: {cache_pk} (exists={cache_pk.exists()})  # {_MEANING.get('cache_pk','')}"
            )

        # Effective per-bin experiment params (handles overlap)
        if (args.zmin is not None) or (args.zmax is not None):
            if args.zmin is None or args.zmax is None:
                raise SystemExit("Must provide both --zmin and --zmax")
            import radiofisher.baofisher as rf

            expt_eff = rf.overlapping_expts(
                expt, zlow=float(args.zmin), zhigh=float(args.zmax)
            )
            print(f"\n=== Effective experiment in bin z=[{args.zmin},{args.zmax}] ===")
            for label, items in split_expt(expt_eff):
                _print_section(label, items)

            if args.index:
                _print_indexed(
                    "Effective experiment (indexed)", _flatten(expt_eff), start_index=1
                )

    # Fisher code global knobs worth recording
    try:
        import radiofisher.baofisher as rf

        print("\n=== Fisher code global settings (radiofisher.baofisher) ===")
        _print_section(
            "Numerics",
            [
                ("NSAMP_K", getattr(rf, "NSAMP_K", None)),
                ("NSAMP_U", getattr(rf, "NSAMP_U", None)),
                ("RSD_FUNCTION", getattr(rf, "RSD_FUNCTION", None)),
                ("CAMB_KMAX", getattr(rf, "CAMB_KMAX", None)),
                ("CAMB_EXEC", getattr(rf, "CAMB_EXEC", None)),
            ],
        )
    except Exception as e:
        print(f"\n(note) Could not import baofisher to show global settings: {e}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        # Common when piping to `head`; exit quietly.
        try:
            sys.stdout.close()
        finally:
            raise SystemExit(0)
