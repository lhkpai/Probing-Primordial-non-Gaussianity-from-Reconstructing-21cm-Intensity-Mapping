"""9.16: retained-mode thermal voxel variance and cylindrical Fisher forecasts.

The notebook is the user-facing entry point.  This module keeps the numerical
implementation importable for verification and avoids rerunning CLASS for every
redshift bin.  All wavenumbers here are comoving Mpc^-1; temperatures are mK.
"""
from __future__ import annotations

import contextlib
import copy
from dataclasses import asdict, dataclass
import io
import json
from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import PchipInterpolator, RegularGridInterpolator
from classy import Class

ROOT = Path(__file__).resolve().parent
BAO_ROOT = ROOT.parent / "bao21cm-master/bao21cm-master"
if str(BAO_ROOT) not in sys.path:
    sys.path.insert(0, str(BAO_ROOT))
from radiofisher import baofisher_k as bf, experiments as ex
from voxel_modes_916 import FiniteBoxVoxelIntegrator

SURVEYS = ["ska_mid1_interferom", "chime_icyl", "hirax_interferom",
           "tianlai_icyl", "meerkat_b1_interferom"]
PARAMETERS = ["lnA", "Omega_m", "omega_b", "omega_cdm", "n_s", "w0", "f_NL"]
FIDUCIAL = dict(lnA=float(np.log(2.1e-9)), Omega_m=.3153, omega_b=.02237,
                omega_cdm=.1200, n_s=.9649, w0=-1., f_NL=0.)


@dataclass(frozen=True)
class Config:
    delta_z: float = .01
    kmax: float = .535                 # Fisher total-k cutoff.
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
    mock_file: str = str(ROOT / "mock_fisher_arr.joblib")
    mock_k_unit: str = "h/Mpc"         # Explicit legacy-product unit assumption.
    # Actual labels in the source paths, not the old nominal linspace(.5,3,6).
    mock_redshifts: tuple = (.5110, .9890, 1.5010, 2.0050, 2.5010, 3.0170)
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
                    cosm, str(ROOT / "transfer_bf_cache.dat"),
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


class MockVariance:
    def __init__(self, cfg):
        data = joblib.load(cfg.mock_file)
        self.z = np.asarray(cfg.mock_redshifts, dtype=float)
        if len(data) != len(self.z):
            raise ValueError("Mock redshift metadata does not match the layer count.")
        if cfg.mock_k_unit not in ("h/Mpc", "1/Mpc"):
            raise ValueError("Explicit mock_k_unit must be h/Mpc or 1/Mpc.")
        scale = cfg.h_mock if cfg.mock_k_unit == "h/Mpc" else 1.
        self.p = np.asarray(data[0]["kper_bin"], float)*scale
        self.ell = np.asarray(data[0]["kpar_bin"], float)*scale
        for d in data:
            if not (np.array_equal(d["kper_bin"], data[0]["kper_bin"])
                    and np.array_equal(d["kpar_bin"], data[0]["kpar_bin"])):
                raise ValueError("Mock layers must use the same two-dimensional axes.")
        cube = np.stack([d["fisher_arr"] for d in data]).astype(float)
        if not np.all(np.isfinite(cube) & (cube >= 0)):
            raise ValueError("Mock variance contains invalid or negative values.")
        for axis in (self.z, self.p, self.ell):
            if np.any(np.diff(axis) <= 0):
                raise ValueError("Mock axes must be strictly increasing.")
        self.interp = RegularGridInterpolator((self.z, self.p, self.ell), cube,
                                              bounds_error=True)

    def evaluate(self, p, ell, z):
        return self.interp(np.column_stack((np.full(p.size, z), p.ravel(), ell.ravel()))
                           ).reshape(p.shape)


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


def cylindrical_grid(e, geom, cfg, mock=None):
    """Gauss quadrature in p and positive ell, with exact spherical upper edge."""
    ellmin, cap = bf.noise_k_limits(geom, e)
    pmin, pmax = transverse_bounds(e, geom)
    ellcap = cap
    if mock is not None:
        pmin, pmax = max(pmin, mock.p[0]), min(pmax, mock.p[-1])
        ellmin, ellcap = max(ellmin, mock.ell[0]), min(ellcap, mock.ell[-1])
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
    x, w = leggauss(cfg.n_parallel)
    ell = ellmin+(upper[:, None]-ellmin)*(x[None, :]+1)/2
    weight = (wp*p*p)[:, None]*(upper[:, None]-ellmin)*w[None, :]/2
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


def fisher_at_z(models, e, z, cfg, mock=None):
    geom = models.fid.geometry(z)
    grid = cylindrical_grid(e, geom, cfg, mock)
    zeros = np.zeros((len(PARAMETERS), len(PARAMETERS)))
    if grid is None:
        return zeros, zeros.copy(), 0.
    p, ell, weights, valid = grid
    pn, _ = noise_power(p, ell, geom, e)
    ps, derivatives = models.power_and_derivatives(p, ell, z)
    # Invalid powers never enter an arithmetic product or an inverse variance.
    d = derivatives[:, valid]
    denom = (ps[valid]+pn[valid])**2
    w = weights[valid]/(4*np.pi**2)
    analytic = (d*(w/denom)) @ d.T
    combined = analytic.copy()
    if mock is not None:
        extra = mock.evaluate(p, ell, z)[valid]*ps[valid]**2
        combined = (d*(w/(denom+extra))) @ d.T
    return analytic, combined, float(weights.sum())


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


def run_survey(name, cfg, models, integrator, mock):
    e = experiment(name, cfg)
    requested, band = redshift_bounds(e)
    edges = bin_edges(*requested, cfg.delta_z)
    matrices = {key: np.zeros((7,7)) for key in ("analytic_full", "analytic_common", "analytic_mock")}
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
            # Clip the redshift integration to data support, never clip a query
            # redshift to the boundary layer. Both compared Fishers share this domain.
            if cfg.run_fisher and mock is not None:
                ml, mh = max(obslo,mock.z[0]), min(obshi,mock.z[-1])
                if mh>ml:
                    zz, ww = gauss_interval(ml,mh,cfg.n_z)
                    for z, weight in zip(zz,ww):
                        local = dict(e,dnutot=e["nu_line"]/(1+ml)-e["nu_line"]/(1+mh))
                        fa, fm, _ = fisher_at_z(models,local,z,cfg,mock)
                        dv = weight*models.fid.volume_element(z,e)
                        matrices["analytic_common"] += dv*fa
                        matrices["analytic_mock"] += dv*fm
        rows.append(row)
        if index % 10 == 0 or index == len(edges)-2:
            print(f"[{name}] bin {index+1}/{len(edges)-1}: {row['status']}", flush=True)
    frame = pd.DataFrame(rows)
    frame["k_max_Mpc_inv"] = cfg.voxel_kmax
    frame["voxel_side_Mpc"] = cfg.boxsize_mpc_h/cfg.grid_size/cfg.h_mock
    frame["t_obs_hours"] = cfg.t_obs_hours
    frame["baseline_source"] = e["baseline_source"]
    frame["variance_estimator"] = "periodic_box_mode_sum"
    prior = planck_fisher()
    constraints = pd.DataFrame(index=PARAMETERS)
    if cfg.run_fisher:
        for label, key in [("sigma_ana_full","analytic_full"),
                           ("sigma_ana_common","analytic_common"),
                           ("sigma_ana_mock","analytic_mock")]:
            constraints[label] = sigma_from_fisher(matrices[key])
            constraints[label+"_cmb"] = sigma_from_fisher(matrices[key]+prior)
    metadata = dict(config=asdict(cfg), requested_z_bounds=requested, band_z_bounds=band,
                    baseline_source=e["baseline_source"],
                    baseline_has_zero_transitions=e["baseline_has_zero_transitions"],
                    variance_definition="Finite periodic-box mode sum; cube-averaged thermal field; original foreground/FoV/baseline masks; no Cfg.",
                    no_extra_nyquist_cut=True,
                    boxsize_Mpc=cfg.boxsize_mpc_h/cfg.h_mock,
                    fundamental_k_Mpc_inv=integrator.fundamental_k,
                    redshift_statistic="volume-weighted average of finite snapshots at configured Gauss nodes",
                    continuous_edge_integral_may_diverge=e["baseline_has_zero_transitions"],
                    mock_mode_normalization="deferred by user; extra denominator = fisher_arr * P_HI^2",
                    mock_redshift_source="variance_audit_20260915/mock_paths.json source filename labels",
                    mock_outside_support="excluded from both compared Fishers; full analytic separately saved")
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out/f"voxel_noise_{name}.csv",index=False)
    joblib.dump(dict(metadata=metadata,rows=rows,variance_table=frame),out/f"voxel_noise_{name}.joblib")
    if cfg.run_fisher:
        constraints.index.name="param"
        constraints.to_csv(out/f"constraints_{name}.csv")
        joblib.dump(dict(parameters=PARAMETERS,matrices=matrices,metadata=metadata),
                    out/f"fisher_{name}.joblib")
    return dict(telescope=name,variance_table=frame,constraints=constraints,
                matrices=matrices,metadata=metadata)


def run_all(cfg=None, surveys=None):
    cfg = cfg or Config()
    surveys = SURVEYS if surveys is None else surveys
    integrator = FiniteBoxVoxelIntegrator(cfg.boxsize_mpc_h,cfg.grid_size,cfg.h_mock,cfg.voxel_kmax)
    mock = MockVariance(cfg) if cfg.run_fisher else None
    models = SignalModels(cfg)
    try:
        results = [run_survey(name,cfg,models,integrator,mock) for name in surveys]
    finally:
        models.close()
    out = Path(cfg.output_dir)
    pd.concat([r["variance_table"] for r in results],ignore_index=True).to_csv(
        out/"voxel_noise_all_surveys.csv",index=False)
    (out/"run_config.json").write_text(json.dumps(asdict(cfg),indent=2),encoding="utf-8")
    return results


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
