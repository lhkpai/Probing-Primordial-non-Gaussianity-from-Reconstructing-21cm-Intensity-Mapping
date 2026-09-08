#!/usr/bin/env python3
"""Run a minimal RadioFisher forecast and print parameter uncertainties.

This script is intended for quick, local runs in this workspace:
- avoids requiring an external CAMB binary by generating `cache_pk.dat` using
  the Python `camb` package (if needed)
- runs in serial (no mpi4py)

Example:
  python3 run_fisher_sigmas.py --expt SKA1MIDbase1 --zmin 0.9 --zmax 1.1

Notes:
- The underlying Fisher calculation can be expensive. Use --nsamp-k/--nsamp-u
  and a narrow z-bin to iterate quickly.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def generate_cache_pk_with_camb(
    *, cosmo: dict, out_path: Path, npoints: int, max_k_mpc_inv: float
) -> None:
    """Generate a z=0 linear matter power spectrum cache file.

    Writes two columns: k [Mpc^-1], P(k) [Mpc^3].
    """
    import camb

    h = float(cosmo["h"])
    omb = float(cosmo["omega_b_0"])
    omm = float(cosmo["omega_M_0"])

    ombh2 = omb * h**2
    omch2 = (omm - omb) * h**2

    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=100.0 * h,
        ombh2=ombh2,
        omch2=omch2,
        mnu=float(cosmo.get("mnu", 0.0)),
        omk=float(1.0 - cosmo["omega_M_0"] - cosmo["omega_lambda_0"]),
    )
    pars.set_dark_energy(w=float(cosmo.get("w0", -1.0)), wa=float(cosmo.get("wa", 0.0)))

    # Start with a reasonable As and then rescale to match sigma8.
    ns = float(cosmo["ns"])
    As0 = 2.1e-9
    pars.InitPower.set_params(As=As0, ns=ns)

    # camb expects kmax in units of h/Mpc for get_matter_power_spectrum inputs.
    max_k_hmpc = max_k_mpc_inv / h
    pars.set_matter_power(redshifts=[0.0], kmax=max_k_hmpc)

    results = camb.get_results(pars)
    sigma8_now = float(results.get_sigma8()[0])
    sigma8_target = float(cosmo.get("sigma_8", sigma8_now))

    # Rescale As to match sigma8 (sigma8 ~ sqrt(As)).
    if sigma8_now > 0 and sigma8_target > 0:
        As = As0 * (sigma8_target / sigma8_now) ** 2
        pars.InitPower.set_params(As=As, ns=ns)
        results = camb.get_results(pars)

    kh, _z, pk = results.get_matter_power_spectrum(
        minkh=1e-4, maxkh=max_k_hmpc, npoints=npoints
    )

    # Convert from (h/Mpc, (Mpc/h)^3) to (1/Mpc, Mpc^3)
    k = np.asarray(kh) * h
    pk0 = np.asarray(pk)[0] / h**3

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Add a dummy header. (The original RadioFisher cache includes a hash.)
    header = "dummyhash#"
    np.savetxt(out_path, np.column_stack([k, pk0]), header=header)


def ensure_cosmo_power_spectrum(*, rf, cosmo: dict, cache_pk: Path) -> dict:
    """Attach pk_nobao(k) and fbao(k) to the cosmo dict.

    This bypasses the external CAMB-binary path in `rf.load_power_spectrum`.
    """
    dat = np.genfromtxt(cache_pk).T
    if dat.shape[0] < 2:
        raise ValueError(f"{cache_pk} does not look like a 2-column k,Pk file")
    k_in, pk_in = dat[0], dat[1]

    cosmo = dict(cosmo)
    cosmo["pk_nobao"], cosmo["fbao"] = rf.spline_pk_nobao(k_in, pk_in)
    cosmo["k_in_max"] = float(np.max(k_in))
    cosmo["k_in_min"] = float(np.min(k_in))
    return cosmo


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expt",
        default="SKA1MIDbase1",
        help="Experiment name from radiofisher.experiments",
    )
    parser.add_argument(
        "--cache-pk", default="cache_pk.dat", help="Path to k,Pk cache file"
    )
    parser.add_argument(
        "--zmin", type=float, default=None, help="Optional single-bin zmin"
    )
    parser.add_argument(
        "--zmax", type=float, default=None, help="Optional single-bin zmax"
    )
    parser.add_argument(
        "--dz",
        type=float,
        default=0.2,
        help="Bin width (used if --zmin/--zmax not set)",
    )
    parser.add_argument("--kmin", type=float, default=1e-3)
    parser.add_argument("--kmax", type=float, default=0.5)
    parser.add_argument("--nsamp-k", type=int, default=180)
    parser.add_argument("--nsamp-u", type=int, default=400)
    parser.add_argument("--pk-npoints", type=int, default=2048)
    parser.add_argument(
        "--pk-kmax",
        type=float,
        default=5.0,
        help="Max k to generate for cache_pk.dat, in Mpc^-1",
    )
    args = parser.parse_args()

    # Use a non-interactive backend for headless environments.
    os.environ.setdefault("MPLBACKEND", "Agg")

    import radiofisher.baofisher as rf
    import radiofisher.experiments as experiments

    expt = getattr(experiments, args.expt)
    cosmo0 = experiments.cosmo

    # Speed/accuracy knobs.
    rf.NSAMP_K = int(args.nsamp_k)
    rf.NSAMP_U = int(args.nsamp_u)

    cache_pk = Path(args.cache_pk)
    if not cache_pk.exists():
        print(f"cache file not found; generating: {cache_pk}")
        generate_cache_pk_with_camb(
            cosmo=cosmo0,
            out_path=cache_pk,
            npoints=args.pk_npoints,
            max_k_mpc_inv=args.pk_kmax,
        )

    cosmo = ensure_cosmo_power_spectrum(rf=rf, cosmo=cosmo0, cache_pk=cache_pk)
    cosmo_fns = rf.background_evolution_splines(cosmo, zmax=10.0)

    # Decide bins.
    if args.zmin is not None or args.zmax is not None:
        if args.zmin is None or args.zmax is None:
            raise ValueError("Must supply both --zmin and --zmax")
        zs = np.array([args.zmin, args.zmax])
    else:
        zs, _zc = rf.zbins_equal_spaced(expt, dz=float(args.dz))

    Ftot = None
    names_ref = None

    for i in range(len(zs) - 1):
        zmin, zmax = float(zs[i]), float(zs[i + 1])
        print(f"\n=== Fisher bin {i}: z=[{zmin:.3f},{zmax:.3f}] ===")
        F, names = rf.fisher(
            zmin,
            zmax,
            cosmo,
            expt,
            cosmo_fns,
            kmin=float(args.kmin),
            kmax=float(args.kmax),
        )

        if names_ref is None:
            names_ref = list(names)
            Ftot = np.array(F, dtype=float)
        else:
            if list(names) != names_ref:
                raise ValueError(
                    "Parameter name order changed across bins; cannot sum matrices safely"
                )
            Ftot += F

    # Invert Fisher matrix -> covariance.
    cov = np.linalg.pinv(Ftot)
    sig = np.sqrt(np.diag(cov))

    print("\n=== Marginalised 1-sigma constraints (sqrt(diag(F^-1))) ===")
    for name, s in zip(names_ref, sig):
        print(f"{name:>12s}  {s:.6e}")

    # Save outputs.
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    stem = f"{args.expt}_z{zs[0]:.3f}-{zs[-1]:.3f}_k{args.kmin:g}-{args.kmax:g}_Nk{args.nsamp_k}_Nu{args.nsamp_u}"
    np.savetxt(out_dir / f"{stem}_F.txt", Ftot, header="# " + " ".join(names_ref))
    np.savetxt(
        out_dir / f"{stem}_sigmas.txt",
        np.column_stack([np.arange(len(sig)), sig]),
        header="# idx sigma",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
