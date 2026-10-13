"""9.16: retained-mode thermal voxel variance and cylindrical Fisher forecasts.

The notebook is the user-facing entry point.  This module keeps the numerical
implementation importable for verification and avoids rerunning CLASS for every
redshift bin.  All wavenumbers here are comoving Mpc^-1; temperatures are mK.
"""
from __future__ import annotations

import contextlib
import copy
from dataclasses import asdict, dataclass
import hashlib
import io
import inspect
import json
from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import PchipInterpolator, RectBivariateSpline
from classy import Class

ROOT = Path(__file__).resolve().parent
RESEARCH_ROOT = ROOT.parent
BAO_ROOT = RESEARCH_ROOT.parent / "bao21cm-master/bao21cm-master"
NOISE_ROOT = Path("/mnt/xiaodong/luantch/LSS21cm_freeloadXD/PosiRestore/FisherMess/noise_hkl")
if str(BAO_ROOT) not in sys.path:
    sys.path.insert(0, str(BAO_ROOT))
from radiofisher import baofisher_k as bf, experiments as ex
from voxel_modes_916 import FiniteBoxVoxelIntegrator

SURVEYS = ["ska_mid1_interferom", "chime_icyl", "hirax_interferom",
           "tianlai_icyl", "meerkat_b1_interferom"]
PARAMETERS = ["lnA", "Omega_m", "omega_b", "omega_cdm", "n_s", "w0", "f_NL"]
FIDUCIAL = dict(lnA=float(np.log(2.1e-9)), Omega_m=.3153, omega_b=.02237,
                omega_cdm=.1200, n_s=.9649, w0=-1., f_NL=0.)
ANALYTIC_DENOMINATOR = "P_HI**2"
TOTAL_DENOMINATOR = "P_HI**2 + P_HI**2 * A_noise_conditional"


@dataclass(frozen=True)
class Config:
    delta_z: float = .01
    kmax: float = .363                 # Fisher total-k cutoff; matches voxel default.
    voxel_kmax: float = .363           # Independent total-k cutoff for voxel noise.
    t_obs_hours: float = 100000.
    boxsize_mpc_h: float = 3000.
    grid_size: int = 512               # Defines voxel averaging width, not a k cutoff.
    h_mock: float = .6766
    n_perp: int = 96
    n_parallel: int = 64
    n_z: int = 2                       # Gauss nodes per observed redshift interval.
    pk_nodes: int = 900
    run_fisher: bool = True
    noise_file: str = str(NOISE_ROOT / "data/noise_hkl_N200.h5")
    noise_wedge: bool = True
    noise_fisher_only: bool = False   # Full valid 200x200 grid; True selects the 39-bin subset.
    output_dir: str = str(ROOT / "outputs_916")

    def __post_init__(self):
        if min(self.delta_z, self.kmax, self.voxel_kmax, self.t_obs_hours, self.h_mock) <= 0:
            raise ValueError("Bin width, kmax, time and h_mock must be positive.")
        if min(self.n_perp, self.n_parallel, self.n_z, self.pk_nodes) < 1:
            raise ValueError("Quadrature sizes must be positive.")


def experiment(name, cfg):
    definitions = {
        "ska_mid1_interferom": (ex.SKA1MIDbase1, "interferom", {}),
        "chime_icyl": (ex.CHIME, "icyl", {}),
        "hirax_interferom": (ex.HIRAX, "interferom", {"Dmin": 6., "Dmax": 200.}),
        "tianlai_icyl": (ex.TIANLAI, "icyl", {"Ncyl": 8, "cyl_area": 1500.}),
        "meerkat_b1_interferom": (ex.MeerKATb1, "interferom", {}),
    }
    original, mode, extra = definitions[name]
    e = copy.deepcopy(original)
    e.update(extra)
    e.update(mode=mode, ttot=cfg.t_obs_hours * 3600. * 1e6, kmax=cfg.kmax)
    path = BAO_ROOT / e["n(x)"]
    if path.is_file():
        e["n(x)"] = bf.load_interferom_file(str(path))
        e["baseline_source"] = str(path)
        density = np.asarray(e["n(x)"].y)
        e["baseline_has_zero_transitions"] = bool(np.any(
            (density[:-1] == 0) != (density[1:] == 0)))
    else:
        e.pop("n(x)")
        e["baseline_source"] = "uniform; assumed Dmin=6 m, Dmax=200 m"
        e["baseline_has_zero_transitions"] = False
        warnings.warn(f"{name}: baseline file missing; {e['baseline_source']}")
    return e


def redshift_bounds(e):
    lo = e["nu_line"] / e["survey_numax"] - 1.
    hi = e["nu_line"] / (e["survey_numax"] - e["survey_dnutot"]) - 1.
    return (lo - .1028, hi + .1917), (lo, hi)


def bin_edges(lo, hi, width):
    """Fixed width, with a short final bin; merge numerical endpoint slivers."""
    edges = lo + width * np.arange(int(np.floor((hi - lo) / width)) + 1)
    if hi - edges[-1] < 1e-4:
        edges[-1] = hi
    else:
        edges = np.r_[edges, hi]
    return edges


def gauss_interval(lo, hi, n):
    x, w = leggauss(n)
    return (lo + hi) / 2 + (hi - lo) * x / 2, w * (hi - lo) / 2


class Cosmology:
    """Existing notebook signal model, evaluated on elementwise k and mu grids."""
    def __init__(self, cfg, params=None):
        self.cfg = cfg
        self.params = FIDUCIAL | (params or {})
        p = self.params
        self.h = np.sqrt((p["omega_b"] + p["omega_cdm"]) / p["Omega_m"])
        self.classy = Class()
        self.classy.set({"output": "mPk", "P_k_max_h/Mpc": 5.,
                         "non linear": "halofit", "z_max_pk": 3.8,
                         "h": self.h, "omega_b": p["omega_b"],
                         "omega_cdm": p["omega_cdm"],
                         "Omega_Lambda": 1 - p["Omega_m"],
                         "A_s": np.exp(p["lnA"]), "n_s": p["n_s"],
                         "w0_fld": p["w0"], "m_ncdm": "0.06", "N_ncdm": 1})
        self.classy.compute()
        self._transfer = None

    def geometry(self, z):
        c = self.classy
        return dict(z=float(z), aperp=1., apar=1., r=c.comoving_distance(z),
                    rnu=(1+z)**2 / c.Hubble(z), ns=self.params["n_s"])

    def volume(self, lo, hi, e):
        return e["Sarea"] * (self.classy.comoving_distance(hi)**3
                             - self.classy.comoving_distance(lo)**3) / 3.

    def volume_element(self, z, e):
        return e["Sarea"] * self.classy.comoving_distance(z)**2 / self.classy.Hubble(z)

    def inverse_transfer(self, k):
        if self._transfer is None:
            p = self.params
            cosm = dict(h=self.h, omega_M_0=p["Omega_m"],
                        omega_b_0=p["omega_b"]/self.h**2,
                        omega_lambda_0=1-p["Omega_m"], ns=p["n_s"],
                        w0=p["w0"], wa=0., mnu=.06, sigma_8=.8, fNL=0.)
            with contextlib.redirect_stdout(io.StringIO()):
                self._transfer = bf.deriv_transfer(
                    cosm, str(RESEARCH_ROOT / "transfer_bf_cache.dat"),
                    kmax=bf.CAMB_KMAX, kref=.002*self.h, force=False)[0]
        return self._transfer(k)

    def signal(self, p, ell, z, fnl_derivative=False):
        k = np.hypot(p, ell)
        mu2 = (ell / k)**2
        # A smooth one-dimensional CLASS interpolation avoids one CLASS call
        # per two-dimensional quadrature node. Validate its accuracy separately.
        kg = np.geomspace(max(float(k.min())*.999, 1e-6), float(k.max())*1.001,
                          self.cfg.pk_nodes)
        pk = PchipInterpolator(np.log(kg), np.log(
            [self.classy.pk(float(q), float(z)) for q in kg]))
        matter = np.exp(pk(np.log(k)))
        tb = .055919 + .23242*z - .024136*z*z
        bias = bf.bias_HI(z, {"bHI0": ex.cosmo.get("bHI0", .677)})
        growth = self.classy.scale_independent_growth_factor_f(z)
        response = bias + growth*mu2
        base = tb**2 * matter * np.exp(-49.*ell**2)
        if fnl_derivative:
            alpha = ((100*self.h/299792.458)**2 * self.params["Omega_m"] * 1.686
                     * self.inverse_transfer(k)
                     / self.classy.scale_independent_growth_factor(z))
            return 2*base*response*3*(bias-1)*alpha
        return base*response**2

    def close(self):
        self.classy.struct_cleanup()
        self.classy.empty()


class SignalModels:
    """Cache CLASS cosmologies; the finite-difference steps match old 9.16."""
    def __init__(self, cfg):
        self.fid = Cosmology(cfg)
        self.variations = {}
        if cfg.run_fisher:
            for par in PARAMETERS[:-1]:
                step = .025*abs(FIDUCIAL[par]) or 1e-6
                self.variations[par] = (
                    Cosmology(cfg, {par: FIDUCIAL[par]+step}),
                    Cosmology(cfg, {par: FIDUCIAL[par]-step}), step)

    def power_and_derivatives(self, p, ell, z):
        power = self.fid.signal(p, ell, z)
        derivs = [(up.signal(p, ell, z)-dn.signal(p, ell, z))/(2*step)
                  for up, dn, step in self.variations.values()]
        derivs.append(self.fid.signal(p, ell, z, fnl_derivative=True))
        return power, np.asarray(derivs)

    def close(self):
        self.fid.close()
        for up, dn, _ in self.variations.values():
            up.close()
            dn.close()


class NoiseCalibration:
    """Thin adapter to the existing HKL reader; Fisher coordinates are Mpc^-1.

    The reader forms A = N_cal * tf_var / (2 * tf_mean**2) at native bins,
    then interpolates A. Never repeat that mode-count conversion here.
    """

    def __init__(self, cfg):
        # Keep the calibration package read-only and reuse its validated reader.
        if str(NOISE_ROOT) not in sys.path:
            sys.path.insert(0, str(NOISE_ROOT))
        from noise_hkl import NoiseHKLTable

        self.table = NoiseHKLTable(cfg.noise_file)
        self.h = cfg.h_mock
        self.wedge = cfg.noise_wedge
        self.fisher_only = cfg.noise_fisher_only
        self.z = self.table.z
        mask = self.table.fisher_mask if self.fisher_only else self.table.N_cal > 0
        ip, il = np.where(mask)
        self.p = self.table.k_perp[np.unique(ip)] * self.h
        self.ell = self.table.k_para[np.unique(il)] * self.h

    def supports(self, z, sigma_mK):
        """Check the actual (z, log sigma) convex hull, without extrapolation."""
        try:
            self.table.node_weights(z, sigma_mK)
        except ValueError:
            return False
        return True

    def _native_k(self, p, ell):
        """Convert Mpc^-1 to h/Mpc; snap only endpoint round-trip roundoff."""
        kp = np.asarray(np.asarray(p, dtype=float)/self.h)
        ka = np.asarray(np.abs(np.asarray(ell, dtype=float))/self.h)
        tolerance = 8*np.finfo(float).eps
        for values, axis in [(kp, self.table.k_perp), (ka, self.table.k_para)]:
            for edge in axis[[0, -1]]:
                values[np.abs(values-edge) <= tolerance*abs(edge)] = edge
        return kp, ka

    def evaluate(self, p, ell, z, sigma_mK, *, bounds_error=False):
        kp, ka = self._native_k(p, ell)
        return self.table.evaluate(
            z, sigma_mK, kp, ka,
            wedge=self.wedge, fisher_only=self.fisher_only,
            bounds_error=bounds_error,
        )

    def evaluate_amplitude(self, p, ell, z, sigma_mK, *, bounds_error=False,
                           method="linear"):
        """Fisher-only A query at one (z, sigma), with the reader's exact rules.

        Both interpolation steps are linear, so first combine native k planes
        at this (z, sigma), then interpolate in k. This avoids repeating the
        same redshift interpolation for hundreds of thousands of modes.
        Every positive-weight node and corner must still be valid. ``both``
        also evaluates a cubic spline on the same normalized A plane. Cubic
        values that are negative/nonfinite explicitly fall back to linear;
        they never remove modes or silently enter a Fisher denominator.
        """
        if method not in {"linear", "spline", "both"}:
            raise ValueError("Noise interpolation must be linear, spline, or both.")
        if method != "linear" and self.fisher_only:
            raise ValueError("Cubic comparison requires the full Noise table (fisher_only=False).")
        p, ell = np.broadcast_arrays(p, ell)
        try:
            node = self.table.node_weights(z, sigma_mK)
        except ValueError:
            if bounds_error:
                raise
            missing = {"A_noise_conditional": np.full(p.shape, np.nan),
                       "valid": np.zeros(p.shape, dtype=bool)}
            if method != "linear":
                missing.update(A_noise_spline=np.full(p.shape, np.nan),
                               spline_fallback=np.zeros(p.shape, dtype=bool))
            return missing
        amplitude = np.zeros_like(self.table.N_cal)
        supported = (self.table.fisher_mask.copy() if self.fisher_only
                     else np.ones_like(amplitude, dtype=bool))
        for (iz, isig), weight in zip(node["indices"], node["weights"]):
            plane = self.table.at_node(int(iz), int(isig), wedge=self.wedge)
            supported &= plane["valid"]
            amplitude += weight * np.where(plane["valid"], plane["A_noise_conditional"], 0.)

        kp, ka = self._native_k(p.ravel(), ell.ravel())
        axes = (self.table.k_perp, self.table.k_para)
        valid = (np.isfinite(kp) & np.isfinite(ka)
                 & (kp >= axes[0][0]) & (kp <= axes[0][-1])
                 & (ka >= axes[1][0]) & (ka <= axes[1][-1]))
        indices, fractions = [], []
        for axis, values in zip(axes, (kp, ka)):
            values = np.where(valid, values, axis[0])
            index = np.clip(np.searchsorted(axis, values, side="right")-1, 0, len(axis)-2)
            fraction = (values-axis[index])/(axis[index+1]-axis[index])
            fraction[np.abs(fraction) < 1e-12] = 0.
            fraction[np.abs(fraction-1) < 1e-12] = 1.
            indices.append(index)
            fractions.append(fraction)
        ix, iy = indices
        fx, fy = fractions
        result = np.zeros_like(kp)
        for dx, dy, weight in [(0, 0, (1-fx)*(1-fy)), (1, 0, fx*(1-fy)),
                               (0, 1, (1-fx)*fy), (1, 1, fx*fy)]:
            corner_valid = supported[ix+dx, iy+dy]
            valid &= corner_valid | (weight == 0)
            result += weight * np.where(corner_valid, amplitude[ix+dx, iy+dy], 0.)
        cubic, fallback = None, None
        if method != "linear":
            # The current full 200x200 calibration is a complete rectangle.
            # Do not fill masked holes to make another dataset fit a spline.
            if (not supported.all() or not np.isfinite(amplitude).all()
                    or np.any(amplitude < 0)):
                raise ValueError("Spline needs a complete, finite, nonnegative calibration plane.")
            cubic = np.full_like(result, np.nan)
            if valid.any():
                spline = RectBivariateSpline(*axes, amplitude, kx=3, ky=3, s=0)
                cubic[valid] = spline.ev(kp[valid], ka[valid])
            fallback = valid & (~np.isfinite(cubic) | (cubic < 0))
            cubic[fallback] = result[fallback]
        if bounds_error and not valid.all():
            raise ValueError("Noise query outside valid calibration support.")
        result[~valid] = np.nan
        answer = {"A_noise_conditional": result.reshape(p.shape),
                  "valid": valid.reshape(p.shape)}
        if cubic is not None:
            answer.update(A_noise_spline=cubic.reshape(p.shape),
                          spline_fallback=fallback.reshape(p.shape))
            if method == "spline":
                answer["A_noise_conditional"] = answer["A_noise_spline"]
        return answer

    def close(self):
        self.table.close()


def transverse_bounds(e, geom):
    """Original file/uniform/FoV envelope; zero-density holes use the exact mask."""
    nu = e["nu_line"]/(1+geom["z"])
    wavelength = 300./nu
    if callable(e.get("n(x)")):
        umin, umax = np.asarray(e["n(x)"].x)[[0, -1]]*nu
    else:
        umin, umax = e["Dmin"]/wavelength, e["Dmax"]/wavelength
    if "cyl" not in e["mode"]:
        umin = max(umin, e["Ddish"]/wavelength)
    return 2*np.pi*np.array([umin, umax])/geom["r"]


def cylindrical_grid(e, geom, cfg, noise=None):
    """Gauss quadrature in p and positive ell, with exact spherical upper edge."""
    ellmin, cap = bf.noise_k_limits(geom, e)
    pmin, pmax = transverse_bounds(e, geom)
    ellcap = cap
    if noise is not None:
        pmin, pmax = max(pmin, noise.p[0]), min(pmax, noise.p[-1])
        ellmin, ellcap = max(ellmin, noise.ell[0]), min(ellcap, noise.ell[-1])
    pmax = min(pmax, np.sqrt(max(0., cap**2-ellmin**2)))
    if pmax <= pmin or ellcap <= ellmin:
        return None
    # Split at every baseline interpolation knot. A single global rule can
    # miss the cylinder's zero-density edge even with hundreds of nodes.
    # The physical measure stays p dp = p^2 d(log p).
    edges = np.geomspace(pmin,pmax,max(8,cfg.n_perp//8)+1)
    if callable(e.get("n(x)")):
        nu = e["nu_line"]/(1+geom["z"])
        knots = np.asarray(e["n(x)"].x)*2*np.pi*nu/geom["r"]
        edges = np.unique(np.r_[edges,knots[(knots>pmin)&(knots<pmax)]])
    if noise is not None:
        # Calibration masks change at bin centres; align quadrature segments.
        edges = np.unique(np.r_[edges, noise.p[(noise.p > pmin) & (noise.p < pmax)]])
    nodes, node_weights = [], []
    order = max(4,int(np.ceil(cfg.n_perp/24)))
    for left,right in zip(edges[:-1],edges[1:]):
        if callable(e.get("n(x)")):
            midx = np.sqrt(left*right)*geom["r"]/(2*np.pi*nu)
            if float(e["n(x)"](midx)) <= 1/bf.INF_NOISE:
                continue
        xpart,wpart = gauss_interval(np.log(left),np.log(right),order)
        nodes.append(xpart)
        node_weights.append(wpart)
    if not nodes:
        return None
    logp, wp = np.concatenate(nodes),np.concatenate(node_weights)
    p = np.exp(logp)
    upper = np.minimum(np.sqrt(np.maximum(0., cap**2-p*p)), ellcap)
    parallel_edges = np.array([ellmin, ellcap])
    if noise is not None:
        parallel_edges = np.unique(np.r_[parallel_edges,
            noise.ell[(noise.ell > ellmin) & (noise.ell < ellcap)]])
    order_parallel = max(4, int(np.ceil(cfg.n_parallel / (len(parallel_edges)-1))))
    x, w = leggauss(order_parallel)
    parallel_nodes, parallel_weights = [], []
    for left, right in zip(parallel_edges[:-1], parallel_edges[1:]):
        width = np.maximum(0., np.minimum(upper, right)-left)
        parallel_nodes.append(left + width[:, None]*(x[None, :]+1)/2)
        parallel_weights.append((wp*p*p)[:, None]*width[:, None]*w[None, :]/2)
    ell = np.concatenate(parallel_nodes, axis=1)
    weight = np.concatenate(parallel_weights, axis=1)
    p = np.broadcast_to(p[:, None], ell.shape)
    valid = bf.interferometer_mode_mask(geom["r"]*p, geom["rnu"]*ell, geom, e)
    return p, ell, np.where(valid, weight, 0.), valid


def noise_power(p, ell, geom, e):
    valid = bf.interferometer_mode_mask(geom["r"]*p, geom["rnu"]*ell, geom, e)
    power = np.full(np.broadcast_shapes(np.shape(p), np.shape(ell)), np.nan)
    p, ell = np.broadcast_arrays(p, ell)
    if np.any(valid):
        with contextlib.redirect_stdout(io.StringIO()):
            cn = bf.Cnoise(geom["r"]*p[valid], geom["rnu"]*ell[valid], geom, e)
        values = cn*geom["r"]**2*geom["rnu"]
        if not np.all(np.isfinite(values) & (values > 0)):
            raise FloatingPointError("Non-finite thermal power inside explicit valid domain.")
        power[valid] = values
    return power, valid


def voxel_variance(integrator, e, geom):
    # Keep the voxel cutoff independent of the Fisher experiment settings.
    e = dict(e, kmax=integrator.kmax)
    kfg, _ = bf.noise_k_limits(geom, e)
    p = integrator.k_perp
    # Evaluate on the first retained lattice plane, away from a roundoff-prone
    # foreground boundary. Undo only its separable channel response below.
    reference = np.ceil(kfg/integrator.fundamental_k)*integrator.fundamental_k
    ellref = np.full_like(p, reference)
    power, valid = noise_power(p, ellref, geom, e)
    sig = np.sqrt(16*np.log(2))*e["nu_line"]/(e["dnu"]*geom["rnu"])
    amplitude = np.zeros_like(p)
    amplitude[valid] = power[valid]*np.exp(-(reference/sig)**2)
    return integrator.integrate(amplitude, sig, kfg, valid_perp=valid)


def _fisher_at_z(models, e, z, cfg, noise=None, sigma_mK=None, *, compare_spline=False):
    """Compute all requested denominators on one grid and one set of derivatives."""
    if noise is not None and (sigma_mK is None or not np.isfinite(sigma_mK) or sigma_mK <= 0):
        raise ValueError("A finite positive voxel RMS is required for Noise calibration.")
    geom = models.fid.geometry(z)
    zeros = np.zeros((len(PARAMETERS), len(PARAMETERS)))
    keys = ["analytic_common", "analytic_noise"]
    if compare_spline:
        if noise is None:
            raise ValueError("Spline comparison requires Noise calibration.")
        keys.append("analytic_noise_spline")
    matrices = {key: zeros.copy() for key in keys}
    full_grid = cylindrical_grid(e, geom, cfg)
    coverage = dict(status="no_instrument_modes", sigma_mK=sigma_mK,
                    instrument_mode_weight=0., supported_mode_weight=0.,
                    supported_fraction=0., candidate_nodes=0, supported_nodes=0)
    if full_grid is None:
        return matrices, coverage
    coverage["instrument_mode_weight"] = float(full_grid[2].sum())
    if noise is not None and not noise.supports(z, sigma_mK):
        coverage["status"] = "outside_z_sigma_support"
        return matrices, coverage
    grid = full_grid if noise is None else cylindrical_grid(e, geom, cfg, noise)
    if grid is None:
        coverage["status"] = "no_k_overlap"
        return matrices, coverage
    p, ell, weights, valid = grid
    valid = valid & (weights > 0)
    coverage["candidate_nodes"] = int(valid.sum())
    if noise is not None:
        query = noise.evaluate_amplitude(p, ell, z, sigma_mK,
                                         method="both" if compare_spline else "linear")
        a_noise = np.asarray(query["A_noise_conditional"])
        valid &= query["valid"] & np.isfinite(a_noise) & (a_noise >= 0)
    if not np.any(valid):
        coverage["status"] = "no_valid_noise_modes"
        return matrices, coverage
    coverage.update(status="ok", supported_nodes=int(valid.sum()),
                    supported_mode_weight=float(weights[valid].sum()))
    coverage["supported_fraction"] = (
        coverage["supported_mode_weight"] / coverage["instrument_mode_weight"])
    # Evaluate only supported modes; NaNs from missing calibration never enter F.
    p_hi, derivatives = models.power_and_derivatives(p[valid], ell[valid], z)
    if not np.all(np.isfinite(p_hi) & (p_hi > 0)):
        raise FloatingPointError("Signal-only Fisher requires finite positive HI power.")
    # Instrumental noise selects the calibration RMS; the analytic term is CV only.
    denom_ana = p_hi**2
    w = weights[valid]/(4*np.pi**2)
    analytic = (derivatives*(w/denom_ana)) @ derivatives.T
    combined = analytic.copy()
    if noise is not None:
        # The calibrated conditional Noise supplies the additional variance.
        denom_noise = p_hi**2 * a_noise[valid]
        denom_total = denom_ana + denom_noise
        combined = (derivatives*(w/denom_total)) @ derivatives.T
        ratio = denom_noise/denom_ana
        coverage.update(A_noise_min=float(a_noise[valid].min()),
                        A_noise_max=float(a_noise[valid].max()),
                        noise_to_analytic_median=float(np.median(ratio)),
                        noise_to_analytic_p90=float(np.quantile(ratio, .9)))
    matrices.update(analytic_common=analytic, analytic_noise=combined)
    if compare_spline:
        a_spline = query["A_noise_spline"][valid]
        fallback = query["spline_fallback"][valid]
        denominator = denom_ana + p_hi**2 * a_spline
        matrices["analytic_noise_spline"] = (derivatives*(w/denominator)) @ derivatives.T
        relative = np.divide(a_spline-a_noise[valid], a_noise[valid],
                             out=np.zeros_like(a_spline), where=a_noise[valid] > 0)
        coverage.update(
            spline_fallback_quadrature_nodes=int(fallback.sum()),
            spline_fallback_mode_weight=float(weights[valid][fallback].sum()),
            spline_max_abs_relative_A_change=float(np.max(np.abs(relative))),
            A_spline_min=float(a_spline.min()), A_spline_max=float(a_spline.max()))
    return matrices, coverage


def fisher_at_z(models, e, z, cfg, noise=None, sigma_mK=None):
    """Original linear-interpolation API; preserve existing callers and results."""
    matrices, coverage = _fisher_at_z(models, e, z, cfg, noise, sigma_mK)
    return matrices["analytic_common"], matrices["analytic_noise"], coverage


def sigma_from_fisher(matrix):
    """Invert a diagonally scaled positive Fisher matrix; do not hide null modes."""
    diag = np.diag(matrix)
    if not np.all(np.isfinite(matrix)) or np.any(diag <= 0):
        return np.full(len(diag), np.nan)
    scale = np.sqrt(diag)
    corr = matrix/scale[:, None]/scale[None, :]
    corr = (corr+corr.T)/2
    eig = np.linalg.eigvalsh(corr)
    if eig.min() <= 1e-12:
        warnings.warn("Fisher is singular/indefinite; marginalized errors are undefined.")
        return np.full(len(diag), np.nan)
    covariance = np.linalg.inv(corr)/scale[:, None]/scale[None, :]
    return np.sqrt(np.diag(covariance))


def planck_fisher():
    # The existing notebook's prior, in PARAMETERS order.
    return np.array([
        [1.3814e4,-3.2729e1,1.8767e5,4.9029e2,6.7237e3,-9.1151e3,0],
        [-3.2729e1,1.1054e8,4.1059e7,-2.3063e8,-1.4401e6,-9.1987e6,0],
        [1.8767e5,4.1059e7,1.3256e9,-2.5375e8,1.7445e7,-2.1510e7,0],
        [4.9029e2,-2.3063e8,-2.5375e8,5.0301e8,8.3734e5,2.1457e7,0],
        [6.7237e3,-1.4401e6,1.7445e7,8.3734e5,5.7721e5,-1.6066e5,0],
        [-9.1151e3,-9.1987e6,-2.1510e7,2.1457e7,-1.6066e5,1.0292e6,0],
        [0,0,0,0,0,0,3.8447e-2]])


def _noise_fisher(models, e, frame, cfg, noise, *, compare_spline=False):
    """Use the supplied voxel-variance table to select physical noise levels.

    Interpolate variance versus z_mid, then take sqrt, matching the HKL
    production convention. Clip the integration interval to tabulated support;
    never use np.interp's endpoint extension as calibration data.
    """
    keys = ["analytic_common", "analytic_noise"]
    if compare_spline:
        keys.append("analytic_noise_spline")
    matrices = {key: np.zeros((len(PARAMETERS), len(PARAMETERS))) for key in keys}
    good = np.isfinite(frame.var_noise_mK2) & (frame.var_noise_mK2 > 0)
    samples = frame.loc[good, ["z_mid", "var_noise_mK2"]].sort_values("z_mid")
    sample_z, sample_var = samples.to_numpy().T
    records = []
    observed_volume = 0.
    for index, row in enumerate(frame.itertuples(index=False)):
        record = dict(telescope=row.telescope, z_low=row.z_low, z_high=row.z_high,
                      z=np.nan, sigma_mK=np.nan, volume_weight_Mpc3=0.,
                      instrument_mode_weight=0., supported_mode_weight=0.,
                      supported_fraction=0., candidate_nodes=0, supported_nodes=0)
        if row.status == "out_of_band":
            records.append(record | {"status": "out_of_band"})
            continue
        observed_volume += models.fid.volume(row.z_obs_low, row.z_obs_high, e)
        if not len(sample_z):
            records.append(record | {"status": "no_voxel_noise_samples"})
            continue
        lo = max(row.z_obs_low, noise.z[0], sample_z[0])
        hi = min(row.z_obs_high, noise.z[-1], sample_z[-1])
        if hi <= lo:
            records.append(record | {"status": "outside_redshift_support"})
            continue
        zs, wz = gauss_interval(lo, hi, cfg.n_z)
        local = dict(e, dnutot=e["nu_line"]/(1+lo)-e["nu_line"]/(1+hi))
        for z, weight in zip(zs, wz):
            sigma_mK = float(np.sqrt(np.interp(z, sample_z, sample_var)))
            local_matrices, coverage = _fisher_at_z(
                models, local, z, cfg, noise, sigma_mK, compare_spline=compare_spline)
            dv = weight * models.fid.volume_element(z, e)
            for key in matrices:
                matrices[key] += dv * local_matrices[key]
            records.append(record | coverage | dict(z=float(z), volume_weight_Mpc3=float(dv)))
        if (index + 1) % 10 == 0 or index == len(frame) - 1:
            print(f"[{row.telescope}] Noise bin {index+1}/{len(frame)}", flush=True)
    coverage = pd.DataFrame(records)
    queried_volume = float(coverage.volume_weight_Mpc3.sum())
    supported_volume = float(coverage.loc[coverage.status == "ok", "volume_weight_Mpc3"].sum())
    instrument_modes = float((coverage.volume_weight_Mpc3 * coverage.instrument_mode_weight).sum())
    supported_modes = float((coverage.volume_weight_Mpc3 * coverage.supported_mode_weight).sum())
    summary = dict(
        status="ok" if supported_modes > 0 else "no_supported_modes",
        observed_volume_Mpc3=float(observed_volume),
        queried_volume_Mpc3=queried_volume,
        supported_volume_fraction=supported_volume / observed_volume if observed_volume else 0.,
        supported_mode_fraction_queried_z=supported_modes / instrument_modes if instrument_modes else 0.,
        status_counts={str(k): int(v) for k, v in coverage.status.value_counts().items()},
    )
    if compare_spline:
        fallback_weight = float((coverage.volume_weight_Mpc3
                                 * coverage.get("spline_fallback_mode_weight", 0.)).sum())
        summary.update(
            spline_fallback_quadrature_nodes=int(
                coverage.get("spline_fallback_quadrature_nodes", pd.Series(dtype=float)).sum()),
            spline_fallback_mode_weight_fraction=(fallback_weight/supported_modes
                                                  if supported_modes else 0.),
            spline_max_abs_relative_A_change=float(
                coverage.get("spline_max_abs_relative_A_change", pd.Series([0.])).max()))
    return matrices, coverage, summary


def noise_fisher(models, e, frame, cfg, noise):
    """Original linear-interpolation forecast; spline comparison is opt-in."""
    matrices, coverage, summary = _noise_fisher(models, e, frame, cfg, noise)
    return matrices["analytic_common"], matrices["analytic_noise"], coverage, summary


def run_survey(name, cfg, models, integrator, noise):
    e = experiment(name, cfg)
    requested, band = redshift_bounds(e)
    edges = bin_edges(*requested, cfg.delta_z)
    matrices = {key: np.zeros((7,7)) for key in ("analytic_full", "analytic_common", "analytic_noise")}
    rows = []
    for index, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        obslo, obshi = max(lo, band[0]), min(hi, band[1])
        row = dict(telescope=name, z_low=lo, z_high=hi, z_mid=(lo+hi)/2,
                   z_obs_low=obslo if obshi>obslo else np.nan,
                   z_obs_high=obshi if obshi>obslo else np.nan,
                   z_effective=np.nan, coverage_fraction=0., var_noise_mK2=np.nan,
                   rms_noise_mK=np.nan, supported_mode_fraction=np.nan,
                   effective_mode_count=np.nan, status="out_of_band")
        if obshi > obslo:
            row["coverage_fraction"] = models.fid.volume(obslo,obshi,e)/models.fid.volume(lo,hi,e)
            row["status"] = "partial_band" if obslo>lo or obshi<hi else "ok"
            zs, wz = gauss_interval(obslo, obshi, cfg.n_z)
            numerator = volume = zeff = modes = fraction = 0.
            for z, weight in zip(zs, wz):
                geom = models.fid.geometry(z)
                local = dict(e, dnutot=e["nu_line"]/(1+obslo)-e["nu_line"]/(1+obshi))
                dv = weight*models.fid.volume_element(z,e)
                result = voxel_variance(integrator, local, geom)
                numerator += dv*result["variance_mK2"]
                modes += dv*result["mode_count"]
                fraction += dv*result["supported_mode_fraction"]
                volume += dv
                zeff += dv*z
                if cfg.run_fisher:
                    fa, _, _ = fisher_at_z(models, local, z, cfg)
                    matrices["analytic_full"] += dv*fa
            row.update(z_effective=zeff/volume, var_noise_mK2=numerator/volume,
                       rms_noise_mK=np.sqrt(numerator/volume),
                       effective_mode_count=modes/volume,
                       supported_mode_fraction=fraction/volume)
            representative = models.fid.geometry(row["z_effective"])
            lower, upper = transverse_bounds(e, representative)
            kfg, _ = bf.noise_k_limits(representative, e)
            row.update(kperp_min_envelope_Mpc_inv=lower,
                       kperp_max_envelope_Mpc_inv=min(upper,np.sqrt(max(0.,cfg.voxel_kmax**2-kfg**2))),
                       kpar_min_Mpc_inv=kfg,
                       kpar_min_lattice_Mpc_inv=np.ceil(kfg/integrator.fundamental_k)*integrator.fundamental_k)
        rows.append(row)
        if index % 10 == 0 or index == len(edges)-2:
            print(f"[{name}] bin {index+1}/{len(edges)-1}: {row['status']}", flush=True)
    frame = pd.DataFrame(rows)
    frame["k_max_Mpc_inv"] = cfg.voxel_kmax
    frame["voxel_side_Mpc"] = cfg.boxsize_mpc_h/cfg.grid_size/cfg.h_mock
    frame["t_obs_hours"] = cfg.t_obs_hours
    frame["baseline_source"] = e["baseline_source"]
    frame["variance_estimator"] = "periodic_box_mode_sum"
    noise_coverage = pd.DataFrame()
    coverage_summary = dict(status="fisher_disabled")
    if cfg.run_fisher:
        fa, fn, noise_coverage, coverage_summary = noise_fisher(models, e, frame, cfg, noise)
        matrices["analytic_common"] = fa
        matrices["analytic_noise"] = fn
    prior = planck_fisher()
    constraints = pd.DataFrame(index=PARAMETERS)
    if cfg.run_fisher:
        for label, key in [("sigma_ana_full","analytic_full"),
                           ("sigma_ana_common","analytic_common"),
                           ("sigma_ana_noise","analytic_noise")]:
            constraints[label] = sigma_from_fisher(matrices[key])
            # Empty HI support must not masquerade as a successful CMB-only forecast.
            constraints[label+"_cmb"] = (
                sigma_from_fisher(matrices[key]+prior) if np.any(np.diag(matrices[key]) > 0)
                else np.full(len(PARAMETERS), np.nan))
        if coverage_summary["status"] == "ok" and not np.isfinite(constraints["sigma_ana_common"]).all():
            coverage_summary["status"] = "singular_fisher"
    metadata = dict(config=asdict(cfg), requested_z_bounds=requested, band_z_bounds=band,
                    baseline_source=e["baseline_source"],
                    baseline_has_zero_transitions=e["baseline_has_zero_transitions"],
                    variance_definition="Finite periodic-box mode sum; cube-averaged thermal field; original foreground/FoV/baseline masks; no Cfg.",
                    no_extra_nyquist_cut=True,
                    boxsize_Mpc=cfg.boxsize_mpc_h/cfg.h_mock,
                    fundamental_k_Mpc_inv=integrator.fundamental_k,
                    redshift_statistic="volume-weighted average of finite snapshots at configured Gauss nodes",
                    continuous_edge_integral_may_diverge=e["baseline_has_zero_transitions"],
                    noise_coverage=coverage_summary)
    if noise is not None:
        metadata.update(
            noise_source_file=str(Path(cfg.noise_file).resolve()),
            noise_fingerprint=noise.table.metadata["scientific_fingerprint"],
            noise_seed_count=int(noise.table.metadata["seed_count"]),
            noise_A_definition="N_cal * tf_var / (2 * tf_mean**2), at native bins before interpolation",
            noise_mode_normalization="Already in A; no further N_cal, N_obs or seed-count factor",
            noise_sigma_source="Saved voxel variance; linear in z_mid, then square root",
            noise_interpolation="Delaunay linear in (z, log sigma); bilinear in (k_perp, k_parallel)",
            fisher_analytic_denominator=ANALYTIC_DENOMINATOR,
            noise_total_denominator=TOTAL_DENOMINATOR,
            noise_outside_support="Excluded from both common-domain Fishers; full analytic saved separately",
            noise_model_limitations=[
                "Analytic covariance is cosmic variance only; calibrated conditional Noise represents the added scatter",
                "Matching voxel RMS does not reproduce anisotropic instrumental noise power",
                "Mode-count transfer assumes approximate 1/N scaling and neglects inter-bin covariance",
                "Recovered f_NL response is approximated by the existing signal derivative",
            ],
        )
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out/f"voxel_noise_{name}.csv",index=False)
    joblib.dump(dict(metadata=metadata,rows=rows,variance_table=frame),out/f"voxel_noise_{name}.joblib")
    if cfg.run_fisher:
        constraints.index.name="param"
        constraints.to_csv(out/f"constraints_{name}.csv")
        joblib.dump(dict(parameters=PARAMETERS,matrices=matrices,metadata=metadata),
                    out/f"fisher_{name}.joblib")
        noise_coverage.to_csv(out/f"noise_coverage_{name}.csv", index=False)
        print(f"[{name}] Noise coverage: {coverage_summary['status']}; "
              f"mode fraction={coverage_summary['supported_mode_fraction_queried_z']:.3g}", flush=True)
    return dict(telescope=name,variance_table=frame,constraints=constraints,
                matrices=matrices,metadata=metadata,noise_coverage=noise_coverage)


def run_all(cfg=None, surveys=None):
    cfg = cfg or Config()
    surveys = SURVEYS if surveys is None else surveys
    integrator = FiniteBoxVoxelIntegrator(cfg.boxsize_mpc_h,cfg.grid_size,cfg.h_mock,cfg.voxel_kmax)
    noise = NoiseCalibration(cfg) if cfg.run_fisher else None
    models = None
    try:
        models = SignalModels(cfg)
        results = [run_survey(name,cfg,models,integrator,noise) for name in surveys]
    finally:
        if models is not None:
            models.close()
        if noise is not None:
            noise.close()
    out = Path(cfg.output_dir)
    pd.concat([r["variance_table"] for r in results],ignore_index=True).to_csv(
        out/"voxel_noise_all_surveys.csv",index=False)
    (out/"run_config.json").write_text(json.dumps(asdict(cfg),indent=2),encoding="utf-8")
    if cfg.run_fisher:
        pd.concat([r["noise_coverage"] for r in results], ignore_index=True).to_csv(
            out/"noise_coverage_all_surveys.csv", index=False)
        summary = {r["telescope"]: r["metadata"]["noise_coverage"] for r in results}
        (out/"noise_calibration_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return results


def load_results(cfg, surveys=None):
    """Read matching saved forecasts without recalculating voxel noise or CLASS."""
    surveys = SURVEYS if surveys is None else surveys
    out, results = Path(cfg.output_dir), []
    for name in surveys:
        voxel_path = out/f"voxel_noise_{name}.joblib"
        fisher_path = out/f"fisher_{name}.joblib"
        if not voxel_path.is_file() or (cfg.run_fisher and not fisher_path.is_file()):
            raise FileNotFoundError(f"Missing saved results for {name}; run the baseline forecast once.")
        voxel = joblib.load(voxel_path)
        saved = joblib.load(fisher_path) if cfg.run_fisher else {}
        metadata = saved.get("metadata", voxel["metadata"])
        if metadata["config"] != asdict(cfg) or voxel["metadata"]["config"] != asdict(cfg):
            raise ValueError("Saved configuration differs from cfg; update the matching inputs first.")
        if cfg.run_fisher and (metadata.get("fisher_analytic_denominator") != ANALYTIC_DENOMINATOR
                               or metadata.get("noise_total_denominator") != TOTAL_DENOMINATOR):
            raise ValueError("Saved Fisher denominator changed; call run_fisher_only(cfg, surveys).")
        results.append(dict(
            telescope=name, variance_table=voxel["variance_table"], metadata=metadata,
            matrices=saved.get("matrices", {}),
            constraints=(pd.read_csv(out/f"constraints_{name}.csv", index_col="param")
                         if cfg.run_fisher else pd.DataFrame()),
            noise_coverage=(pd.read_csv(out/f"noise_coverage_{name}.csv")
                            if cfg.run_fisher else pd.DataFrame())))
    return results


def _analytic_fisher(models, e, frame, cfg):
    """Integrate the full instrument domain using saved observed z boundaries."""
    matrix = np.zeros((len(PARAMETERS), len(PARAMETERS)))
    for index, row in enumerate(frame.itertuples(index=False)):
        if row.status == "out_of_band":
            continue
        zs, wz = gauss_interval(row.z_obs_low, row.z_obs_high, cfg.n_z)
        local = dict(e, dnutot=e["nu_line"]/(1+row.z_obs_low)
                     - e["nu_line"]/(1+row.z_obs_high))
        for z, weight in zip(zs, wz):
            analytic, _, _ = fisher_at_z(models, local, z, cfg)
            matrix += weight*models.fid.volume_element(z, e)*analytic
        if (index+1) % 20 == 0 or index == len(frame)-1:
            print(f"[{row.telescope}] Analytic Fisher bin {index+1}/{len(frame)}", flush=True)
    return matrix


def _interpolation_result(name, matrices, summary, prior):
    """Build both interpolation forecasts from their shared-domain matrices."""
    constraints = pd.DataFrame(index=PARAMETERS)
    supported = bool(np.any(np.diag(matrices["analytic_common"]) > 0))
    for label, key in [("sigma_ana_common", "analytic_common"),
                       ("sigma_noise_linear", "analytic_noise"),
                       ("sigma_noise_spline", "analytic_noise_spline")]:
        constraints[label] = sigma_from_fisher(matrices[key])
        constraints[label+"_cmb"] = (sigma_from_fisher(matrices[key]+prior)
                                     if supported else np.full(len(PARAMETERS), np.nan))
    constraints.index.name = "param"
    fnl = constraints.loc["f_NL"].to_dict()
    fnl.update(
        telescope=name,
        relative_change_percent=100*(fnl["sigma_noise_spline"]/fnl["sigma_noise_linear"]-1),
        relative_change_cmb_percent=100*(fnl["sigma_noise_spline_cmb"]/fnl["sigma_noise_linear_cmb"]-1),
        spline_fallback_mode_fraction=summary["spline_fallback_mode_weight_fraction"])
    return constraints, fnl


def _save_interpolation_comparison(cfg, surveys, records, saved_surveys):
    """Write one current comparison after the matching baseline has been saved."""
    comparison = pd.DataFrame(records)
    comparison = comparison[["telescope"]+[key for key in comparison if key != "telescope"]]
    out = Path(cfg.output_dir)
    joblib.dump(dict(
        fingerprint=_comparison_fingerprint(cfg, surveys), parameters=PARAMETERS,
        comparison=comparison,
        methods=dict(linear="Delaunay linear in (z, log sigma); bilinear in k",
                     spline="Same (z, log sigma); RectBivariateSpline(kx=3, ky=3, s=0) on native A"),
        normalization="A=N_cal*tf_var/(2*tf_mean**2), before interpolation",
        spline_fallback="Negative/nonfinite cubic values use linear A; same mode mask; no extrapolation",
        denominator=TOTAL_DENOMINATOR, surveys=saved_surveys),
        out/"noise_interpolation_comparison.joblib")
    comparison.to_csv(out/"noise_interpolation_comparison.csv", index=False)
    return comparison


def run_fisher_only(cfg=None, surveys=None, *, reuse_saved=True):
    """Update Fisher and both interpolations; never calculate voxel variances.

    Saved voxel tables supply the calibration RMS. The same common-domain
    quadrature produces the analytic, linear and spline matrices together.
    Matching results are read directly on subsequent notebook executions.
    """
    cfg = cfg or Config()
    surveys = SURVEYS if surveys is None else surveys
    if not cfg.run_fisher or cfg.noise_fisher_only:
        raise ValueError("Fisher-only comparison requires run_fisher=True and the full Noise table.")
    out = Path(cfg.output_dir)
    if reuse_saved:
        try:
            previous = load_results(cfg, surveys)
            cached = joblib.load(out/"noise_interpolation_comparison.joblib")
            if cached.get("fingerprint") == _comparison_fingerprint(cfg, surveys):
                print("Using saved Fisher forecasts; voxel variances are reused.", flush=True)
                return previous
        except (FileNotFoundError, ValueError):
            pass
    voxels = {}
    for name in surveys:
        path = out/f"voxel_noise_{name}.joblib"
        voxel = joblib.load(path)
        if voxel["metadata"]["config"] != asdict(cfg):
            raise ValueError(f"{name}: saved voxel inputs do not match cfg.")
        voxels[name] = voxel
    print("Updating Fisher only: using saved voxel variances and the existing Noise table.", flush=True)
    noise, models = NoiseCalibration(cfg), None
    results, records, saved_surveys = [], [], {}
    prior = planck_fisher()
    try:
        models = SignalModels(cfg)
        for name, voxel in voxels.items():
            metadata = copy.deepcopy(voxel["metadata"])
            if metadata["noise_fingerprint"] != noise.table.metadata["scientific_fingerprint"]:
                raise ValueError("Noise calibration changed; verify the saved voxel-to-Noise matching first.")
            frame, e = voxel["variance_table"], experiment(name, cfg)
            full = _analytic_fisher(models, e, frame, cfg)
            matrices, coverage, summary = _noise_fisher(
                models, e, frame, cfg, noise, compare_spline=True)
            compared, fnl = _interpolation_result(name, matrices, summary, prior)
            saved_surveys[name] = dict(
                matrices={key: value for key, value in matrices.items()},
                constraints=compared, coverage_summary=summary,
                baseline_max_scaled_difference=0.)
            matrices["analytic_full"] = full
            constraints = pd.DataFrame(index=PARAMETERS)
            for label, key in [("sigma_ana_full", "analytic_full"),
                               ("sigma_ana_common", "analytic_common"),
                               ("sigma_ana_noise", "analytic_noise"),
                               ("sigma_ana_noise_spline", "analytic_noise_spline")]:
                constraints[label] = sigma_from_fisher(matrices[key])
                constraints[label+"_cmb"] = (
                    sigma_from_fisher(matrices[key]+prior) if np.any(np.diag(matrices[key]) > 0)
                    else np.full(len(PARAMETERS), np.nan))
            constraints.index.name = "param"
            metadata.update(
                fisher_analytic_denominator=ANALYTIC_DENOMINATOR,
                noise_total_denominator=TOTAL_DENOMINATOR, noise_coverage=summary,
                voxel_variance_reused=True,
                voxel_source_sha256={str(out/f"voxel_noise_{name}.{suffix}"):
                    hashlib.sha256((out/f"voxel_noise_{name}.{suffix}").read_bytes()).hexdigest()
                    for suffix in ["csv", "joblib"]},
                noise_sigma_source="Saved voxel variance; linear in z_mid, then square root",
                noise_model_limitations=[
                    "Analytic covariance is cosmic variance only; conditional Noise represents the added scatter",
                    "Matching voxel RMS does not reproduce anisotropic instrumental noise power",
                    "Mode-count transfer assumes approximate 1/N scaling and neglects inter-bin covariance",
                    "Recovered f_NL response is approximated by the existing signal derivative"])
            results.append(dict(telescope=name, variance_table=frame, metadata=metadata,
                                matrices=matrices, constraints=constraints, noise_coverage=coverage))
            records.append(fnl)
            print(f"[{name}] f_NL: analytic={fnl['sigma_ana_common']:.6g}, "
                  f"linear={fnl['sigma_noise_linear']:.6g}, spline={fnl['sigma_noise_spline']:.6g}", flush=True)
    finally:
        if models is not None:
            models.close()
        noise.close()
    # Update Fisher outputs together. Voxel CSV/joblib files remain untouched.
    for result in results:
        name = result["telescope"]
        result["constraints"].to_csv(out/f"constraints_{name}.csv")
        joblib.dump(dict(parameters=PARAMETERS, matrices=result["matrices"],
                         metadata=result["metadata"]), out/f"fisher_{name}.joblib")
        result["noise_coverage"].to_csv(out/f"noise_coverage_{name}.csv", index=False)
    pd.concat([r["noise_coverage"] for r in results], ignore_index=True).to_csv(
        out/"noise_coverage_all_surveys.csv", index=False)
    (out/"noise_calibration_summary.json").write_text(json.dumps(
        {r["telescope"]: r["metadata"]["noise_coverage"] for r in results}, indent=2))
    _save_interpolation_comparison(cfg, surveys, records, saved_surveys)
    return results


def _comparison_fingerprint(cfg, surveys):
    """Invalidate comparison results when inputs or relevant calculation code change."""
    out = Path(cfg.output_dir)
    paths = [out/f"{kind}_{name}.joblib" for name in surveys
             for kind in ["fisher", "voxel_noise"]]
    paths += [NOISE_ROOT/"noise_hkl.py", Path(bf.__file__), Path(ex.__file__)]
    functions = [Cosmology, SignalModels, NoiseCalibration, experiment,
                 cylindrical_grid, noise_power, _fisher_at_z, _noise_fisher,
                 sigma_from_fisher, planck_fisher, _analytic_fisher,
                 _interpolation_result, _save_interpolation_comparison,
                 run_fisher_only, compare_noise_interpolations]
    source = "\n".join(inspect.getsource(function) for function in functions)
    stat = Path(cfg.noise_file).stat()
    return dict(
        schema=2, config=asdict(cfg), surveys=list(surveys),
        analytic_denominator=ANALYTIC_DENOMINATOR, total_denominator=TOTAL_DENOMINATOR,
        code_sha256=hashlib.sha256(source.encode()).hexdigest(),
        inputs_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        noise_file_size=stat.st_size, noise_file_mtime_ns=stat.st_mtime_ns)


def compare_noise_interpolations(cfg, surveys=None, *, reuse_saved=True):
    """Compare linear and cubic A interpolation with shared modes and derivatives.

    Reuse saved voxel variances and the full analytic forecast. Only the
    common-domain Fisher integration is evaluated, once for both methods.
    Cache results in two fixed files in the existing output directory.
    """
    if not cfg.run_fisher or cfg.noise_fisher_only:
        raise ValueError("Interpolation comparison requires run_fisher=True and the full Noise table.")
    surveys = SURVEYS if surveys is None else surveys
    baseline = load_results(cfg, surveys)
    fingerprint = _comparison_fingerprint(cfg, surveys)
    out = Path(cfg.output_dir)
    cache_path = out/"noise_interpolation_comparison.joblib"
    csv_path = out/"noise_interpolation_comparison.csv"
    if reuse_saved and cache_path.is_file():
        cached = joblib.load(cache_path)
        if cached.get("fingerprint") == fingerprint:
            cached["comparison"].to_csv(csv_path, index=False)
            print("Using saved linear/spline comparison; no Fisher recalculation.", flush=True)
            return cached["comparison"]

    records, saved_surveys = [], {}
    noise, models = NoiseCalibration(cfg), None
    prior = planck_fisher()
    try:
        models = SignalModels(cfg)
        for previous in baseline:
            name = previous["telescope"]
            if previous["metadata"]["noise_fingerprint"] != noise.table.metadata["scientific_fingerprint"]:
                raise ValueError("Noise calibration changed; update the baseline forecast first.")
            e = experiment(name, cfg)
            matrices, _, summary = _noise_fisher(
                models, e, previous["variance_table"], cfg, noise, compare_spline=True)
            max_change = 0.
            for key in ["analytic_common", "analytic_noise"]:
                reference = previous["matrices"][key]
                scale = np.sqrt(np.maximum(np.diag(reference), np.finfo(float).tiny))
                change = float(np.max(np.abs(
                    (matrices[key]-reference)/scale[:, None]/scale[None, :])))
                max_change = max(max_change, change)
                if change > 1e-8:
                    raise ValueError(f"{name}: baseline changed; update it before comparing interpolation.")

            constraints, fnl = _interpolation_result(name, matrices, summary, prior)
            records.append(fnl)
            saved_surveys[name] = dict(
                matrices=matrices, constraints=constraints, coverage_summary=summary,
                baseline_max_scaled_difference=max_change)
            print(f"[{name}] f_NL: linear={fnl['sigma_noise_linear']:.6g}, "
                  f"spline={fnl['sigma_noise_spline']:.6g}; "
                  f"change={fnl['relative_change_percent']:+.3g}%", flush=True)
    finally:
        if models is not None:
            models.close()
        noise.close()

    return _save_interpolation_comparison(cfg, surveys, records, saved_surveys)


def plot_results(results, output_dir=None):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for result in results:
        d=result["variance_table"]
        good=d["status"]!="out_of_band"
        axes[0].semilogy(d.loc[good,"z_effective"],d.loc[good,"var_noise_mK2"],label=result["telescope"])
        axes[1].plot(d.loc[good,"z_effective"],d.loc[good,"supported_mode_fraction"],label=result["telescope"])
    axes[0].set(xlabel="z",ylabel=r"$\sigma_N^2$ [mK$^2$]")
    axes[1].set(xlabel="z",ylabel="Baseline coverage after radial cutoff",ylim=(0,1))
    for ax in axes:
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=8)
    if output_dir:
        fig.savefig(Path(output_dir)/"voxel_noise_all_surveys.png",dpi=160)
    return fig
