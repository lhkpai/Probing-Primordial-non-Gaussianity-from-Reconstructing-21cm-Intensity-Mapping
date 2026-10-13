"""Independent numerical checks for the 9.16 workflow; no science files edited."""
import copy
import contextlib
import io
import json
from pathlib import Path
import sys
from dataclasses import replace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import forecast_916 as f
from voxel_modes_916 import FiniteBoxVoxelIntegrator
from radiofisher import baofisher as original


def check_finite_box():
    integrator = FiniteBoxVoxelIntegrator(80., 10, 1., .6)
    p = integrator.k_perp
    lower, sigma = .1, .9
    support = (p >= .12) & (p <= .51)
    estimate = integrator.integrate(2.+p, sigma, lower, support)
    n = np.arange(-8, 9)
    kf = 2*np.pi/80
    x,y,z = np.meshgrid(n*kf,n*kf,n*kf,indexing='ij')
    perp = np.hypot(x,y)
    valid = (np.sqrt(x*x+y*y+z*z)<=.6) & (abs(z)>=lower) & (perp>=.12) & (perp<=.51)
    window = (np.sinc(x*8/(2*np.pi))*np.sinc(y*8/(2*np.pi))*np.sinc(z*8/(2*np.pi)))**2
    expected = np.sum(((2+perp)*np.exp((z/sigma)**2)*window)[valid])/80**3
    np.testing.assert_allclose(estimate['variance_mK2'],expected,rtol=3e-14)
    assert estimate['mode_count'] == int(valid.sum())
    return float(abs(estimate['variance_mK2']/expected-1))


def check_coordinate_measure():
    cap=.535
    p,wp=f.gauss_interval(0,cap,240)
    ellmax=np.sqrt(cap*cap-p*p)
    # Exact ell integral of a constant spectrum; independent k/mu coordinates.
    cylindrical=np.sum(wp*p*ellmax)/(4*np.pi**2)
    k,wk=f.gauss_interval(0,cap,240)
    spherical=np.sum(wk*k*k)/(4*np.pi**2)
    expected=cap**3/(12*np.pi**2)
    np.testing.assert_allclose(spherical,expected,rtol=1e-12)
    np.testing.assert_allclose(cylindrical,expected,rtol=3e-7)
    return float(abs(cylindrical/spherical-1))


def check_noise_spline(noise, cfg):
    """Check normalization, units, bounds, and an actual cubic overshoot."""
    node = noise.table.at_node(6, 9, wedge=cfg.noise_wedge)
    p, ell = np.meshgrid(noise.table.k_perp*cfg.h_mock,
                         noise.table.k_para*cfg.h_mock, indexing="ij")
    query = noise.evaluate_amplitude(p, ell, node["z"], node["sigma_mK"],
                                     method="both", bounds_error=True)
    expected = node["N_cal"]*node["tf_var"]/(2*node["tf_mean"]**2)
    np.testing.assert_allclose(query["A_noise_conditional"], expected, rtol=2e-14)
    np.testing.assert_allclose(query["A_noise_spline"], expected, rtol=3e-12, atol=1e-14)
    assert not query["spline_fallback"].any()

    # This calibrated low-noise plane has genuine negative cubic overshoots.
    # Fallback preserves the original positive A and the original valid mask.
    table = f.pd.read_csv(Path(cfg.output_dir)/"voxel_noise_tianlai_icyl.csv")
    good = table.var_noise_mK2.notna()
    sigma = float(np.sqrt(np.interp(1., table.loc[good, "z_mid"],
                                   table.loc[good, "var_noise_mK2"])))
    pmid = (noise.p[:-1]+noise.p[1:])/2
    lmid = (noise.ell[:-1]+noise.ell[1:])/2
    p, ell = np.meshgrid(pmid, lmid, indexing="ij")
    query = noise.evaluate_amplitude(p, ell, 1., sigma, method="both", bounds_error=True)
    reference = noise.evaluate(p, ell, 1., sigma, bounds_error=True)
    np.testing.assert_array_equal(query["valid"], reference["valid"])
    np.testing.assert_allclose(query["A_noise_conditional"],
                               reference["A_noise_conditional"], rtol=3e-14)
    fallback = query["spline_fallback"]
    assert fallback.any()
    np.testing.assert_array_equal(query["A_noise_spline"][fallback],
                                  query["A_noise_conditional"][fallback])
    assert np.isfinite(query["A_noise_spline"]).all()
    assert (query["A_noise_spline"] >= 0).all()
    outside = noise.evaluate_amplitude(noise.p[0]/2, noise.ell[0], 1., sigma,
                                       method="both")
    assert not outside["valid"] and np.isnan(outside["A_noise_spline"])
    return dict(spline_native_nodes_checked=expected.size,
                spline_same_valid_domain=True,
                spline_negative_overshoot_fallback_nodes=int(fallback.sum()),
                spline_outside_support_rejected=True)


def run():
    report={'finite_box_bruteforce_relative_error':check_finite_box(),
            'constant_spectrum_coordinate_relative_error':check_coordinate_measure()}
    cfg=f.Config(n_perp=96,n_parallel=64)
    models=f.SignalModels(cfg)
    geom=models.fid.geometry(1.)
    noise=f.NoiseCalibration(cfg)
    integ=FiniteBoxVoxelIntegrator(cfg.boxsize_mpc_h,cfg.grid_size,cfg.h_mock,cfg.voxel_kmax)
    report['surveys']={}
    try:
        report.update(check_noise_spline(noise, cfg))
        for name in f.SURVEYS:
            e=f.experiment(name,cfg);e['dnutot']=100.
            p=np.geomspace(.001,.55,50)[:,None]
            ell=np.linspace(0,.55,60)[None,:]
            p,ell=np.broadcast_arrays(p,ell)
            old=copy.deepcopy(e);old.pop('kmax')
            with contextlib.redirect_stdout(io.StringIO()):
                a=original.Cnoise(geom['r']*p,geom['rnu']*ell,geom,old)
                b=f.bf.Cnoise(geom['r']*p,geom['rnu']*ell,geom,old)
            np.testing.assert_array_equal(a,b)
            variance=f.voxel_variance(integ,e,geom)['variance_mK2']
            ten=copy.deepcopy(e);ten['ttot']*=10
            variance_ten=f.voxel_variance(integ,ten,geom)['variance_mK2']
            np.testing.assert_allclose(variance_ten/variance,.1,rtol=1e-13)
            fa,_,_=f.fisher_at_z(models,e,1.,cfg)
            fine=replace(cfg,n_perp=192,n_parallel=128)
            ff,_,_=f.fisher_at_z(models,e,1.,fine)
            sigma=f.sigma_from_fisher(fa);sigma_fine=f.sigma_from_fisher(ff)
            assert np.isfinite(sigma).all()
            assert np.max(abs(sigma_fine/sigma-1)) < 1e-3
            table=f.pd.read_csv(Path(cfg.output_dir)/f'voxel_noise_{name}.csv')
            good=np.isfinite(table.var_noise_mK2)&(table.var_noise_mK2>0)
            sigma_mK=float(np.sqrt(np.interp(1.,table.loc[good,'z_mid'],table.loc[good,'var_noise_mK2'])))
            common,combined,coverage=f.fisher_at_z(models,e,1.,cfg,noise,sigma_mK)
            # Same-domain covariance increase must not add information.
            scale=np.sqrt(np.maximum(np.diag(common),np.finfo(float).tiny))
            difference=(common-combined)/scale[:,None]/scale[None,:]
            assert np.linalg.eigvalsh((difference+difference.T)/2).min()>-1e-10
            report['surveys'][name]={
                'original_default_noise_identical':True,
                'variance_z1_mK2':variance,
                'tenfold_time_variance_ratio':variance_ten/variance,
                'fisher_refinement_fnl_sigma_relative_change':float(sigma_fine[-1]/sigma[-1]-1),
                'fisher_refinement_max_sigma_relative_change':float(np.max(abs(sigma_fine/sigma-1))),
                'noise_does_not_increase_information':True,
                'noise_support_z1':coverage['status'],
            }
        # Signal interpolation is checked against direct CLASS evaluation.
        p=np.array([.05,.1,.2,.4]);ell=np.array([.02,.07,.1,.2]);k=np.hypot(p,ell)
        signal=models.fid.signal(p,ell,1.)
        c=models.fid.classy;bias=f.bf.bias_HI(1.,{'bHI0':f.ex.cosmo.get('bHI0',.677)})
        tb=.055919+.23242-.024136
        direct=tb**2*(bias+c.scale_independent_growth_factor_f(1.)*(ell/k)**2)**2
        direct*=np.array([c.pk(float(q),1.) for q in k])*np.exp(-49*ell**2)
        report['signal_interpolation_max_relative_error']=float(np.max(abs(signal/direct-1)))
        assert report['signal_interpolation_max_relative_error']<2e-5
        # Native mode normalization and physical/h units, checked against stored statistics.
        node=noise.table.at_node(6,9,wedge=cfg.noise_wedge)
        ix,iy=np.where(node['fisher_mask'])
        kp=noise.table.k_perp[ix]*cfg.h_mock
        ka=noise.table.k_para[iy]*cfg.h_mock
        query=noise.evaluate(kp,ka,node['z'],node['sigma_mK'],bounds_error=True)
        expected=node['N_cal'][ix,iy]*node['tf_var'][ix,iy]/(2*node['tf_mean'][ix,iy]**2)
        np.testing.assert_allclose(query['A_noise_conditional'],expected,rtol=2e-14)
        # A must be normalized at the four original bins before bilinear interpolation.
        kp_mid=np.mean(noise.table.k_perp[1:3])*cfg.h_mock
        ka_mid=np.mean(noise.table.k_para[2:4])*cfg.h_mock
        interpolated=noise.evaluate(kp_mid,ka_mid,node['z'],node['sigma_mK'],bounds_error=True)
        r=node['tf_var'][1:3,2:4]/node['tf_mean'][1:3,2:4]**2
        n=node['N_cal'][1:3,2:4]
        np.testing.assert_allclose(interpolated['A_noise_conditional'],np.mean(n*r/2),rtol=2e-14)
        assert not np.isclose(np.mean(n*r/2),np.mean(n)*np.mean(r)/2,rtol=1e-5)
        report['noise_native_bins_checked']=len(ix)
        report['noise_normalized_before_interpolation']=True
        try:
            noise.evaluate(kp[0]/2,ka[0],node['z'],node['sigma_mK'],bounds_error=True)
        except ValueError:
            report['noise_outside_support_rejected']=True
        else:
            raise AssertionError('Noise extrapolation must not be silently accepted.')
        missing=noise.evaluate(kp[0]/2,ka[0],node['z'],node['sigma_mK'])
        assert not missing['valid'] and np.isnan(missing['A_noise_conditional'])
        assert not noise.supports(node['z'],1e6)
        # The production default uses measured high-k statistics; the original
        # 39-bin option remains available as a low-k diagnostic.
        low_k_noise=f.NoiseCalibration(replace(cfg,noise_fisher_only=True))
        try:
            outside_low_k=low_k_noise.evaluate(.06,.03,node['z'],node['sigma_mK'])
            high_k=noise.evaluate(.06,.03,node['z'],node['sigma_mK'],bounds_error=True)
            assert not outside_low_k['valid']
            assert high_k['valid'] and np.isfinite(high_k['A_noise_conditional'])
            report['noise_full_table_high_k_supported']=True
            # The optimized fixed-(z,sigma) query must agree with the original
            # reader at exact bins, between bins, and outside either support.
            rng=np.random.default_rng(916)
            checks=0
            max_error=0.
            for calibration in [noise,low_k_noise]:
                for iz,isig in [(0,0),(6,9),(17,15)]:
                    z0=calibration.table.z[iz]
                    sigma0=calibration.table.sigma_mK[iz,isig]
                    z1=(z0+calibration.table.z[iz+1])/2
                    sigma1=np.sqrt(sigma0*calibration.table.sigma_mK[iz+1,isig])
                    for zq,sq in [(z0,sigma0),(z1,sigma1),(10.,sigma0)]:
                        pmax=calibration.p[-1]
                        emax=calibration.ell[-1]
                        pp=np.r_[kp, rng.uniform(calibration.p[0],pmax,128),
                                 calibration.p[0]/2, pmax*1.1]
                        ll=np.r_[ka, rng.uniform(calibration.ell[0],emax,128),
                                 calibration.ell[0], emax]
                        ref=calibration.evaluate(pp,ll,zq,sq)
                        fast=calibration.evaluate_amplitude(pp,ll,zq,sq)
                        np.testing.assert_array_equal(fast['valid'],ref['valid'])
                        np.testing.assert_allclose(fast['A_noise_conditional'],
                                                   ref['A_noise_conditional'],rtol=3e-14,equal_nan=True)
                        ok=ref['valid']&(ref['A_noise_conditional']>0)
                        if ok.any():
                            max_error=max(max_error,float(np.max(abs(
                                fast['A_noise_conditional'][ok]/ref['A_noise_conditional'][ok]-1))))
                        checks+=len(pp)
            report['noise_optimized_reader_queries_checked']=checks
            report['noise_optimized_reader_max_relative_difference']=max_error
        finally:
            low_k_noise.close()
        # Exercise nonempty support at a second redshift, independently of
        # the z=1 checks and the calibration's native (z, sigma) nodes.
        name='hirax_interferom'
        table=f.pd.read_csv(Path(cfg.output_dir)/f'voxel_noise_{name}.csv')
        good=np.isfinite(table.var_noise_mK2)&(table.var_noise_mK2>0)
        z=2.501
        sigma_mK=float(np.sqrt(np.interp(z,table.loc[good,'z_mid'],table.loc[good,'var_noise_mK2'])))
        e=dict(f.experiment(name,cfg),dnutot=100.)
        matrices,coverage=f._fisher_at_z(models,e,z,cfg,noise,sigma_mK,compare_spline=True)
        common,combined=matrices['analytic_common'],matrices['analytic_noise']
        assert coverage['supported_nodes']>0
        fine=replace(cfg,n_perp=192,n_parallel=128)
        matrices_fine,_=f._fisher_at_z(models,e,z,fine,noise,sigma_mK,compare_spline=True)
        common_fine,combined_fine=matrices_fine['analytic_common'],matrices_fine['analytic_noise']
        sigma=f.sigma_from_fisher(common)
        sigma_fine=f.sigma_from_fisher(common_fine)
        sigma_noise=f.sigma_from_fisher(combined)
        sigma_noise_fine=f.sigma_from_fisher(combined_fine)
        refinement=max(float(np.max(abs(sigma_fine/sigma-1))),
                       float(np.max(abs(sigma_noise_fine/sigma_noise-1))))
        assert refinement<1e-3
        scale=np.sqrt(np.diag(common))
        difference=(common-combined)/scale[:,None]/scale[None,:]
        assert np.linalg.eigvalsh((difference+difference.T)/2).min()>-1e-10
        assert 0<combined[-1,-1]<common[-1,-1]
        spline=matrices['analytic_noise_spline']
        spline_fine=matrices_fine['analytic_noise_spline']
        spline_refinement=float(np.max(abs(f.sigma_from_fisher(spline_fine)
                                          /f.sigma_from_fisher(spline)-1)))
        assert spline_refinement<1e-3
        difference=(common-spline)/scale[:,None]/scale[None,:]
        assert np.linalg.eigvalsh((difference+difference.T)/2).min()>-1e-10
        report['noise_nonempty_information_check']={
            'telescope':name,'z':z,'supported_nodes':coverage['supported_nodes'],
            'full_grid_refinement_max_sigma_relative_change':refinement,
            'spline_refinement_max_sigma_relative_change':spline_refinement,
            'fnl_diagonal_information_ratio':float(combined[-1,-1]/common[-1,-1])}
    finally:
        models.close()
        noise.close()
    (Path(cfg.output_dir)/'validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


def verify_fisher_only():
    """Verify the new covariance without recalculating any voxel variance."""
    out = ROOT/"outputs_916"
    cfg = f.Config(**json.loads((out/"run_config.json").read_text()))
    previous = json.loads((out/"validation.json").read_text())
    report = {key: value for key, value in previous.items()
              if key not in {"surveys", "noise_nonempty_information_check", "fisher_only"}}
    models, noise = f.SignalModels(cfg), f.NoiseCalibration(cfg)
    try:
        report.update(check_noise_spline(noise, cfg))
        z, name = 2.501, "hirax_interferom"
        table = f.pd.read_csv(out/f"voxel_noise_{name}.csv")
        good = np.isfinite(table.var_noise_mK2) & (table.var_noise_mK2 > 0)
        sigma = float(np.sqrt(np.interp(z, table.loc[good, "z_mid"],
                                       table.loc[good, "var_noise_mK2"])))
        e = dict(f.experiment(name, cfg), dnutot=100.)
        forbidden = AssertionError("Fisher-only validation must not calculate thermal or voxel power.")
        with patch.object(f, "noise_power", side_effect=forbidden), \
             patch.object(f, "voxel_variance", side_effect=forbidden), \
             patch.object(f, "FiniteBoxVoxelIntegrator", side_effect=forbidden):
            matrices, coverage = f._fisher_at_z(
                models, e, z, cfg, noise, sigma, compare_spline=True)
            p, ell, weight, valid = f.cylindrical_grid(e, models.fid.geometry(z), cfg, noise)
            query = noise.evaluate_amplitude(p, ell, z, sigma, method="both")
            valid &= (weight > 0) & query["valid"]
            power, derivatives = models.power_and_derivatives(p[valid], ell[valid], z)
            logarithmic = derivatives/power
            w = weight[valid]/(4*np.pi**2)
            independent = (logarithmic*w) @ logarithmic.T
            scale = np.sqrt(np.diag(independent))
            analytic_error = float(np.max(np.abs(
                (matrices["analytic_common"]-independent)/scale[:, None]/scale[None, :])))
            assert analytic_error < 1e-11
            for key, field in [("analytic_noise", "A_noise_conditional"),
                               ("analytic_noise_spline", "A_noise_spline")]:
                expected = (logarithmic*(w/(1+query[field][valid]))) @ logarithmic.T
                residual = (matrices[key]-expected)/scale[:, None]/scale[None, :]
                assert np.max(np.abs(residual)) < 1e-11
                difference = (independent-matrices[key])/scale[:, None]/scale[None, :]
                assert np.linalg.eigvalsh((difference+difference.T)/2).min() > -1e-10
            np.testing.assert_allclose(coverage["noise_to_analytic_median"],
                                       np.median(query["A_noise_conditional"][valid]), rtol=1e-13)
            fine, _ = f._fisher_at_z(models, e, z,
                replace(cfg, n_perp=192, n_parallel=128), noise, sigma, compare_spline=True)
        refinement = {key: float(np.max(np.abs(
            f.sigma_from_fisher(fine[key])/f.sigma_from_fisher(value)-1)))
            for key, value in matrices.items()}
        assert max(refinement.values()) < 1e-3
        report["fisher_only"] = dict(
            denominator=f.TOTAL_DENOMINATOR, telescope=name, z=z,
            covariance_verified_independently=True,
            analytic_max_scaled_difference=analytic_error,
            voxel_variances_not_calculated=True,
            thermal_power_not_used_in_fisher=True,
            noise_does_not_increase_information=True,
            refinement_max_relative_sigma_change=refinement,
            voxel_validation_reused=True)
    finally:
        models.close()
        noise.close()
    (out/"validation.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report["fisher_only"], indent=2))


if __name__=='__main__':
    if "--fisher-only" in sys.argv:
        verify_fisher_only()
    else:
        run()
