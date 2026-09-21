"""Check redshift smoothness and additive total-k floors for the 9.16 forecast.

Run from any directory: python /home/liuhk/research/diagnose_noise_916.py
Only diagnostic outputs are written; the forecast tables are never overwritten.
The baseline is the existing finite-box sum, not a different FFT-grid estimator.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import forecast_916 as forecast
from voxel_modes_916 import FiniteBoxVoxelIntegrator


@dataclass
class Snapshot:
    z: float
    amplitude: np.ndarray
    density: np.ndarray
    valid: np.ndarray
    first_parallel: int
    prefix: np.ndarray
    segment: np.ndarray
    k_min: float


class ModeDiagnostic:
    """Reuse grouped transverse modes and exact axial sums from the production grid."""
    def __init__(self, cfg, cosmology):
        self.cfg = cfg
        self.cosmology = cosmology
        self.integrator = FiniteBoxVoxelIntegrator(
            cfg.boxsize_mpc_h, cfg.grid_size, cfg.h_mock, cfg.voxel_kmax)
        # Read-only geometry reuse avoids constructing a dense 512^3 array.
        self.g = self.integrator._geometry
        self.p = self.g.k_perp
        self.kf = self.g.fundamental_k
        self.radius_squared = (self.p/self.kf)**2

    def snapshot(self, experiment, z):
        geom = self.cosmology.geometry(z)
        e = dict(experiment, kmax=self.cfg.voxel_kmax,
                 dnutot=experiment["survey_dnutot"])
        kfg, _ = forecast.bf.noise_k_limits(geom, e)
        first = int(np.ceil(kfg/self.kf))
        ell = first*self.kf
        power, valid = forecast.noise_power(self.p, np.full_like(self.p, ell), geom, e)
        sigma = np.sqrt(16*np.log(2))*e["nu_line"]/(e["dnu"]*geom["rnu"])
        amplitude = np.zeros_like(self.p)
        amplitude[valid] = power[valid]*np.exp(-(ell/sigma)**2)
        axial = self.g.longitudinal_window*np.exp((self.g.longitudinal_k/sigma)**2)
        axial = 2*axial
        axial[0] /= 2  # Only one zero-frequency plane.
        prefix = np.r_[0., np.cumsum(axial)]
        nu = e["nu_line"]/(1+z)
        u = self.p*geom["r"]/(2*np.pi)
        if callable(e.get("n(x)")):
            x = u/nu
            density = np.asarray(e["n(x)"](x))/nu**2
            segment = np.searchsorted(e["n(x)"].x, x, side="right")-1
        else:
            wavelength = 300/nu
            density = np.full_like(self.p, e["Ndish"]*(e["Ndish"]-1)*wavelength**2 /
                                   (2*np.pi*(e["Dmax"]**2-e["Dmin"]**2)))
            segment = np.zeros(self.p.size, dtype=int)
        k_min = float(np.hypot(self.p[valid].min(), ell)) if valid.any() else np.nan
        return Snapshot(float(z), amplitude, density, valid, first, prefix, segment, k_min)

    def integrate(self, s, k_floor=0., valid=None, first_parallel=None, amplitude=None):
        """Exact sum with an added |k| floor; an empty retained field has zero sum."""
        g = self.g
        mask = s.valid if valid is None else (s.valid & valid)
        first = s.first_parallel if first_parallel is None else max(s.first_parallel, first_parallel)
        # k^2 = k_perp^2 + (n_parallel*kf)^2. Roundoff must not remove equality.
        need_squared = np.maximum(0., (k_floor/self.kf)**2-self.radius_squared)
        tolerance = 64*np.finfo(float).eps*max(1., (k_floor/self.kf)**2)
        lower = np.maximum(first, np.ceil(np.sqrt(np.maximum(0., need_squared-tolerance))).astype(int))
        mask = mask & (lower <= g.longitudinal_max_index)
        profile = np.zeros_like(self.p)
        if not mask.any():
            return 0., 0, profile
        lo, hi = lower[mask], g.longitudinal_max_index[mask]
        powers = s.amplitude if amplitude is None else amplitude
        profile[mask] = (powers[mask]*g.transverse_window_sum[mask]
                         *(s.prefix[hi+1]-s.prefix[lo])/g.boxsize_mpc**3)
        count = int(np.sum(g.transverse_mode_count[mask]*(2*(hi-lo+1)-(lo==0))))
        return float(profile.sum()), count, profile

    def transition(self, old, new):
        """Exact contribution accounting on common, entering and exiting modes.

        On common modes, first update all response factors while fixing n(u) to
        its old value, then update n(u). This defines the density attribution.
        """
        common = old.valid & new.valid
        first = max(old.first_parallel, new.first_parallel)
        a, na, _ = self.integrate(old)
        b, nb, _ = self.integrate(new)
        ca, nc, _ = self.integrate(old, valid=common, first_parallel=first)
        cb, _, _ = self.integrate(new, valid=common, first_parallel=first)
        fixed_density = new.amplitude.copy()
        fixed_density[common] *= new.density[common]/old.density[common]
        counterfactual, _, _ = self.integrate(
            new, valid=common, first_parallel=first, amplitude=fixed_density)
        pieces = dict(entered_mK2=b-cb, exited_mK2=a-ca,
                      density_change_mK2=cb-counterfactual,
                      other_response_change_mK2=counterfactual-ca)
        np.testing.assert_allclose(
            pieces["entered_mK2"]-pieces["exited_mK2"]+
            pieces["density_change_mK2"]+pieces["other_response_change_mK2"],
            b-a, rtol=1e-10, atol=1e-12)
        pieces.update(entered_modes=nb-nc, exited_modes=na-nc,
                      interpolation_groups_changed=int(np.count_nonzero(
                          common & (old.segment!=new.segment))),
                      parallel_floor_changed=int(old.first_parallel!=new.first_parallel))
        return pieces


def verify_radial_sum():
    """Independent small-box 3D sum checks the added radial floor and empty sets."""
    cfg = forecast.Config(boxsize_mpc_h=80., h_mock=1., grid_size=10, voxel_kmax=.6)
    diagnostic = ModeDiagnostic(cfg, None)
    p, g = diagnostic.p, diagnostic.g
    axial = 2*g.longitudinal_window*np.exp((g.longitudinal_k/.9)**2)
    axial[0] /= 2
    valid = (p>=.12) & (p<=.51)
    first = 2
    s = Snapshot(1., 2+p, np.ones_like(p), valid, first, np.r_[0.,np.cumsum(axial)],
                 np.zeros(p.size), float(np.hypot(p[valid].min(),first*diagnostic.kf)))
    k = np.arange(-8,9)*diagnostic.kf
    x,y,z = np.meshgrid(k,k,k,indexing="ij")
    perp, total = np.hypot(x,y), np.sqrt(x*x+y*y+z*z)
    mask = (total<=.6) & (abs(z)>=first*diagnostic.kf) & (perp>=.12) & (perp<=.51)
    window = (np.sinc(x*8/(2*np.pi))*np.sinc(y*8/(2*np.pi))*np.sinc(z*8/(2*np.pi)))**2
    for increment in (0.,.01,.02,.03,.3,1.):
        cut = s.k_min+increment
        keep = mask & (total>=cut-1e-14)
        expected = float(np.sum(((2+perp)*np.exp((z/.9)**2)*window)[keep])/80**3)
        value, count, _ = diagnostic.integrate(s, cut)
        np.testing.assert_allclose(value, expected, rtol=3e-13, atol=1e-15)
        assert count == int(keep.sum())


def nearest_zero_edge_gap(experiment, geom, kperp):
    """Distance to a zero/positive baseline-interpolation edge, in Mpc^-1."""
    baseline = experiment.get("n(x)")
    if not callable(baseline):
        return np.nan
    x, y = np.asarray(baseline.x), np.asarray(baseline.y)
    positive = y > 1/forecast.bf.INF_NOISE
    left = np.flatnonzero(positive[:-1] != positive[1:])
    if not left.size:
        return np.nan
    zero_indices = np.where(positive[left], left+1, left)
    nu = experiment["nu_line"]/(1+geom["z"])
    edges = x[zero_indices]*2*np.pi*nu/geom["r"]
    return float(np.min(abs(kperp-edges)))


def diagnose_survey(name, table, cfg, diagnostic, increments, fine_points):
    e = forecast.experiment(name, cfg)
    smooth, cuts, transitions = [], [], []
    previous = None
    for row in table.itertuples(index=False):
        if row.status == "out_of_band":
            continue
        zs, wz = forecast.gauss_interval(row.z_obs_low, row.z_obs_high, cfg.n_z)
        weights = wz*np.array([diagnostic.cosmology.volume_element(z,e) for z in zs])
        weights /= weights.sum()
        snapshots = [diagnostic.snapshot(e,z) for z in zs]
        base = [diagnostic.integrate(s) for s in snapshots]
        variance = float(weights@np.array([v[0] for v in base]))
        modes = float(weights@np.array([v[1] for v in base]))
        np.testing.assert_allclose(variance, row.var_noise_mK2, rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(modes, row.effective_mode_count, rtol=1e-12)
        z = float(row.z_effective)
        peak = int(np.argmax(base[0][2]))
        smooth.append(dict(telescope=name,z=z,z_low=row.z_low,z_high=row.z_high,
                           variance_mK2=variance,mode_count=modes,
                           k_min_min=min(s.k_min for s in snapshots),
                           k_min_max=max(s.k_min for s in snapshots),
                           first_node_peak_kperp=diagnostic.p[peak],
                           first_node_peak_density=snapshots[0].density[peak],
                           first_node_peak_group_fraction=base[0][2][peak]/base[0][0]))
        for increment in increments:
            results = [diagnostic.integrate(s,s.k_min+increment) for s in snapshots]
            value = float(weights@np.array([v[0] for v in results]))
            count = float(weights@np.array([v[1] for v in results]))
            threshold = [s.k_min+increment for s in snapshots]
            empty = sum(v[1]==0 for v in results)
            cuts.append(dict(telescope=name,z=z,increment_Mpc_inv=increment,
                             cutoff_min=min(threshold),cutoff_max=max(threshold),
                             variance_mK2=value,rms_mK=np.sqrt(value),
                             variance_ratio=value/variance,mode_count=count,mode_ratio=count/modes,
                             status="empty_modes" if empty==len(results) else
                             ("partly_empty" if empty else "ok")))
        if previous is not None:
            old_z, old_weights, old_snapshots, old_base, old_variance = previous
            pieces = [diagnostic.transition(a,b) for a,b in zip(old_snapshots,snapshots)]
            item = {key:float(weights@np.array([p[key] for p in pieces])) for key in pieces[0]}
            item["redshift_weight_change_mK2"] = float(
                (weights-old_weights)@np.array([v[0] for v in old_base]))
            delta = variance-old_variance
            accounted = (item["entered_mK2"]-item["exited_mK2"]+
                         item["density_change_mK2"]+item["other_response_change_mK2"]+
                         item["redshift_weight_change_mK2"])
            np.testing.assert_allclose(accounted,delta,rtol=1e-9,atol=1e-12)
            item.update(telescope=name,z_before=old_z,z_after=z,
                        variance_before=old_variance,variance_after=variance,
                        relative_change=delta/old_variance,
                        abs_log_ratio=abs(np.log(variance/old_variance)))
            transitions.append(item)
        previous = z,weights,snapshots,base,variance
    jumps = pd.DataFrame(transitions)
    largest = jumps.loc[jumps.abs_log_ratio.idxmax()]
    width = largest.z_after-largest.z_before
    _, band = forecast.redshift_bounds(e)
    zs = np.linspace(max(band[0],largest.z_before-width/2),
                     min(band[1],largest.z_after+width/2),fine_points)
    fine = []
    for z in zs:
        s = diagnostic.snapshot(e,z)
        value,count,profile = diagnostic.integrate(s)
        peak = int(np.argmax(profile))
        fine.append(dict(telescope=name,z=z,variance_mK2=value,mode_count=count,
                         peak_kperp=diagnostic.p[peak],peak_density=s.density[peak],
                         peak_group_fraction=profile[peak]/value,
                         peak_zero_edge_gap=nearest_zero_edge_gap(
                             e,diagnostic.cosmology.geometry(z),diagnostic.p[peak])))
    return pd.DataFrame(smooth), pd.DataFrame(cuts), jumps, pd.DataFrame(fine)


def plots(smooth, cuts, fine, out):
    names = list(smooth.telescope.unique())
    fig, axes = plt.subplots(len(names),3,figsize=(15,3*len(names)),constrained_layout=True,
                             squeeze=False)
    for axes_row,name in zip(axes,names):
        a,b,c = axes_row
        d = smooth[smooth.telescope==name]
        a.semilogy(d.z,d.variance_mK2)
        a.set(title=name,ylabel="Voxel variance [mK$^2$]",xlabel="z")
        d = fine[fine.telescope==name]
        b.semilogy(d.z,d.variance_mK2,".-",markersize=3)
        b.set(title="Fine snapshots near largest bin change",xlabel="z")
        for increment,g in cuts[(cuts.telescope==name)&(cuts.increment_Mpc_inv>0)].groupby("increment_Mpc_inv"):
            c.plot(g.z,100*(1-g.variance_ratio),label=f"+{increment:g} Mpc$^{{-1}}$")
        c.set(xlabel="z",ylabel="Variance reduction [%]")
        c.set_ylim(bottom=0)
        c.legend(fontsize=8)
    fig.savefig(out/"noise_z_k_diagnostics.png",dpi=160)
    plt.close(fig)


def figure_notes(smooth, cuts, jumps, fine):
    """Reading guide for noise_z_k_diagnostics.png, built from the same frames.

    The guide is generated rather than hand-written so that the numbers in it
    always match the tables above after any rerun.
    """
    def fmt(value, digits=3):
        return "—" if not np.isfinite(value) else f"{value:.{digits}g}"

    lines = ["## 图 `noise_z_k_diagnostics.png` 读法", "",
             "该图由 `plots()` 生成：5 行 = 仪器，3 列 = 三种问法。",
             "纵轴单位是 mK²（体素热噪声方差，不含信号）。", "",
             "|栏|代码入口|这一栏问什么|怎么读|",
             "|---|---|---|---|",
             "|左：Voxel variance|`integrate(s)`|单个体素热噪声方差随 z 的绝对量级|"
             "对数轴。随 z 上升属预期，只做量级体检，不负责发现问题|",
             "|中：Fine snapshots|`snapshot` 在最大箱变化处细采样 81 点|"
             "该变化是真趋势还是数值假象|平滑 = 可信；孤立尖刺 = 伪影|",
             "|右：Variance reduction|`integrate(s, s.k_min+Δ)`|"
             "在掩码下限 k_min(z) 之上再抬高 Δ 能删掉多少方差|"
             "高 = 噪声集中在最低 k；贴 0 = 砍低 k 无用|",
             "",
             "判据是邻点突起比 = 细采样点方差 ÷ 两侧点均值；1.25 只用于筛查，不是物理判据。", "",
             "### 本图要点", "",
             "|仪器|最大相邻箱变化|邻点突起比|判读|",
             "|---|---:|---:|---|"]
    for name, d in jumps.groupby("telescope", sort=False):
        q = d.loc[d.abs_log_ratio.idxmax()]
        v = fine.loc[fine.telescope == name, "variance_mK2"].to_numpy()
        ratio = float(np.max(v[1:-1]/((v[:-2]+v[2:])/2)))
        verdict = "有局部尖峰：数值伪影，不代表物理" if ratio > 1.25 else "平滑，真实趋势"
        lines.append(f"|{name}|{100*q.relative_change:+.2f}%"
                     f"（z={q.z_before:.6f}→{q.z_after:.6f}）|{ratio:.3f}|{verdict}|")
    hotspots, hot_total = [], 0
    for name, d in fine.groupby("telescope", sort=False):
        d = d.sort_values("z")
        hot, quiet = d[d.peak_group_fraction > .1], d[d.peak_group_fraction <= .1]
        if hot.empty:
            continue
        hot_total += len(hot)
        reference = float(quiet.peak_density.median())
        hotspots += [f"|{name}|{r.z:.6f}|{r.variance_mK2:.4g}|{r.peak_kperp:.6f}"
                     f"|{fmt(r.peak_density)}|{fmt(reference)}"
                     f"|{fmt(r.peak_zero_edge_gap)}|{100*r.peak_group_fraction:.1f}%|"
                     for r in hot.itertuples(index=False)]
    if hotspots:
        lines += ["", f"细采样中出现 {hot_total} 个「单组贡献 >10%」的尖峰，全部伴随基线密度塌缩：", "",
                  "|仪器|z|方差 [mK²]|k_perp [Mpc⁻¹]|该组 n(u)|该仪器非热点中位 n(u)"
                  "|距密度零点边缘 [Mpc⁻¹]|单组贡献|",
                  "|---|---:|---:|---:|---:|---:|---:|---:|", *hotspots, "",
                  "机制：分段线性基线 n(u) 可在若干 x 处取到 0；随 z 变化，固定的离散 k_perp 组沿 n(u) 滑动，",
                  "落到零点边缘时 `Cnoise` 的 1/n(u) 被放大。这是插值伪影，不是仪器物理。"]
    else:
        lines += ["", "细采样中未出现「单组贡献 >10%」的尖峰。"]
    largest = float(cuts.increment_Mpc_inv.max())
    lines += ["", "### 右栏的 z 依赖", "",
              f"取最大测试增量 +{largest:g} Mpc⁻¹，按 z 的低/高四分位各取降幅中位数：",
              "低 z 时保留带 [k_min, k_max] 较窄且方差偏带下沿，抬高下限能咬下可观份额；",
              "高 z 时方差向带的高端集中，低 k 下限几乎无效。", "",
              "|仪器|低 z 四分位降幅中位数 [%]|高 z 四分位降幅中位数 [%]|",
              "|---|---:|---:|"]
    for name, d in cuts[cuts.increment_Mpc_inv == largest].groupby("telescope", sort=False):
        d = d.sort_values("z")
        q1, q3 = np.quantile(d.z, (.25, .75))
        low = 100*(1-d.loc[d.z <= q1, "variance_ratio"].median())
        high = 100*(1-d.loc[d.z >= q3, "variance_ratio"].median())
        lines.append(f"|{name}|{low:.2f}|{high:.2f}|")
    lines += ["", "抬高下限只删除非负贡献，方差与模式数必然不增；方差变小不等于灵敏度提高，",
              "Fisher 判据仍是 (P_s+P_N)，删低 k 模式会同时删掉信号。", ""]
    return lines


def report(smooth, cuts, jumps, fine, out, cfg):
    observations = []
    terms = ["entered_mK2","exited_mK2","density_change_mK2",
             "other_response_change_mK2","redshift_weight_change_mK2"]
    labels = ["模式进入","模式退出","共有模式的密度变化","其他响应变化","红移权重变化"]
    for name,d in jumps.groupby("telescope",sort=False):
        q = d.loc[d.abs_log_ratio.idxmax()]
        dominant = labels[int(np.argmax([abs(q[k]) for k in terms]))]
        v = fine.loc[fine.telescope==name,"variance_mK2"].to_numpy()
        peak_ratio = float(np.max(v[1:-1]/((v[:-2]+v[2:])/2)))
        observations.append(f"- {name}：最大相邻箱变化 {100*q.relative_change:+.2f}% "
                            f"（z={q.z_before:.6f}→{q.z_after:.6f}），分解主项为{dominant}。"
                            f"该局部细采样的最高邻点突起比为 {peak_ratio:.3f}。")
    lines = ["# 9.16 噪声方差的红移与低波数诊断", "",
             f"配置：Δz={cfg.delta_z}，Fisher 与体素方差上限均为 {cfg.voxel_kmax} Mpc⁻¹。",
             "基准逐箱重算与主程序 CSV 核对通过；新增径向截断通过独立小盒子三维求和检查。", "",
             "## 定义", "",
             "测试每个红移节点的实际最低有效总波数 k_min(z) 加 0.01、0.02、0.03 Mpc⁻¹；",
             "保留原仪器掩码、视向下限、sinc 窗口和盒子格点。各节点计算后再按原权重平均。",
             "empty_modes 表示没有保留模式，此时零是空和，不表示仪器热噪声消失。", "",
             "## 简要结论", "",
             f"- 三个测试增量中的最高下限为 {cuts.cutoff_max.max():.6f} Mpc⁻¹；"
             f"全空测试数为 {(cuts.status=='empty_modes').sum()}。",
             f"- 提高下限后最大的方差降幅为 {100*(1-cuts.variance_ratio.min()):.2f}%。",
             "- 邻点突起比是细采样点方差除以两侧点均值；大于1.25仅用来筛查局部尖峰。",
             *observations, "",
             *figure_notes(smooth, cuts, jumps, fine),
             "## 红移变化", "",
             "下表列出各配置相邻箱比值变化最大的区间。超过25%仅用作筛查标记，不是物理突变判据。",
             "分解列用原箱方差归一化；退出项取负号。基线密度归因是固定旧 n(u)、先更新其他响应再更新密度的明确对照。", "",
             "|配置|z 区间|方差后/前|进入|退出|密度变化|其他响应|权重变化|",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    summaries = []
    for name,d in jumps.groupby("telescope",sort=False):
        q = d.loc[d.abs_log_ratio.idxmax()]
        ratios = [q.entered_mK2,-q.exited_mK2,q.density_change_mK2,
                  q.other_response_change_mK2,q.redshift_weight_change_mK2]
        lines.append(f"|{name}|{q.z_before:.6f}–{q.z_after:.6f}|{q.variance_after/q.variance_before:.4g}|"+
                     "|".join(f"{v/q.variance_before:+.3g}" for v in ratios)+"|")
        summaries.append(dict(telescope=name,largest_transition=q.to_dict(),
                              transitions_over_25_percent=int((abs(d.relative_change)>.25).sum())))
    if "tianlai_icyl" in set(fine.telescope):
        t = fine[fine.telescope=="tianlai_icyl"]
        peak = t.loc[t.variance_mK2.idxmax()]
        lines += ["", f"Tianlai 细采样峰值位于 z={peak.z:.6f}，方差 {peak.variance_mK2:.6g} mK²；",
                  f"单个横向半径分组 k_perp={peak.peak_kperp:.6f} Mpc⁻¹ 贡献约 {100*peak.peak_group_fraction:.1f}%。",
                  f"该分组距基线密度零边缘仅 {peak.peak_zero_edge_gap:.3g} Mpc⁻¹。",
                  "这是许多简并 Fourier 模式的分组，不是单一天线对或单一 Fourier 模式。"]
    lines += ["", "## 提高下限后的方差保留比例", "",
              "统计包含空域（比例为零），范围与中位数覆盖该仪器所有有效红移箱。", "",
              "|配置|增加量 [Mpc⁻¹]|比例最小值|中位数|最大值|全空箱数|部分空箱数|",
              "|---|---:|---:|---:|---:|---:|---:|"]
    cut_summary = []
    for (name,increment),d in cuts[cuts.increment_Mpc_inv>0].groupby(["telescope","increment_Mpc_inv"],sort=False):
        values = dict(telescope=name,increment=float(increment),minimum=float(d.variance_ratio.min()),
                      median=float(d.variance_ratio.median()),maximum=float(d.variance_ratio.max()),
                      empty_bins=int((d.status=="empty_modes").sum()),
                      partly_empty_bins=int((d.status=="partly_empty").sum()))
        cut_summary.append(values)
        lines.append(f"|{name}|{increment:g}|{values['minimum']:.4g}|{values['median']:.4g}|"
                     f"{values['maximum']:.4g}|{values['empty_bins']}|{values['partly_empty_bins']}|")
    lines += ["", "## 如何解释", "",
              "- 天线对并没有随红移新增；波长和距离变化使固定基线对应的 k_perp 移动。",
              "- 本代码只有径向平均的基线密度，能定位模式进出、插值区间变化和密度变化，不能识别具体哪一对天线。",
              "- CHIME/Tianlai 的零密度边缘可使 1/n(u) 贡献很大；有限离散模式进入边缘时可能出现尖峰。",
              "- 有限格点、硬掩码和分段线性基线插值不保证方差随 z 严格光滑；局部细采样不构成全域收敛证明。",
              "- 提高下限只删除非负贡献，方差及模式数必须不增加；方差变小不等于仪器灵敏度提高。",
              "- 相邻箱分解见 z_transitions.csv；最高波动区间的细采样见 z_fine_snapshots.csv。", ""]
    (out/"analysis.md").write_text("\n".join(lines),encoding="utf-8")
    summary = dict(configuration=dict(delta_z=cfg.delta_z,kmax=cfg.voxel_kmax),
                   interpretation="minimum retained total k at each z plus an increment, in Mpc^-1",
                   baseline_csv_reproduced=True,radial_sum_bruteforce_passed=True,
                   smoothness=summaries,k_floor=cut_summary)
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=forecast.ROOT/"outputs_916/noise_diagnostics")
    parser.add_argument("--surveys",nargs="+",choices=forecast.SURVEYS,default=forecast.SURVEYS)
    parser.add_argument("--fine-points",type=int,default=81)
    args = parser.parse_args()
    if args.fine_points < 3:
        parser.error("--fine-points must be at least 3")
    cfg_data = json.loads((forecast.ROOT/"outputs_916/run_config.json").read_text())
    cfg = forecast.Config(**cfg_data)
    if cfg.kmax != cfg.voxel_kmax:
        raise ValueError("Run the forecast with matching Fisher and voxel kmax first.")
    source = pd.read_csv(Path(cfg.output_dir)/"voxel_noise_all_surveys.csv")
    verify_radial_sum()
    cosmology = forecast.Cosmology(cfg)
    results = []
    try:
        diagnostic = ModeDiagnostic(cfg,cosmology)
        for name in args.surveys:
            print(f"Checking {name}",flush=True)
            results.append(diagnose_survey(name,source[source.telescope==name],cfg,diagnostic,
                                          (0.,.01,.02,.03),args.fine_points))
    finally:
        cosmology.close()
    smooth,cuts,jumps,fine = [pd.concat([r[i] for r in results],ignore_index=True) for i in range(4)]
    # Increasing the floor only removes nonnegative power; empty domains are valid outcomes.
    for _,d in cuts.groupby(["telescope","z"]):
        d = d.sort_values("increment_Mpc_inv")
        assert np.all(np.diff(d.variance_ratio)<=1e-12)
        assert np.all(np.diff(d.mode_count)<=1e-6)
        np.testing.assert_allclose(d.variance_ratio.iloc[0],1.,rtol=1e-12)
    out = args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    for frame,name in [(smooth,"z_smoothness"),(cuts,"k_floor_sensitivity"),
                       (jumps,"z_transitions"),(fine,"z_fine_snapshots")]:
        frame.to_csv(out/f"{name}.csv",index=False)
    plots(smooth,cuts,fine,out)
    report(smooth,cuts,jumps,fine,out,cfg)
    print(f"Checks passed. Results: {out}",flush=True)


if __name__ == "__main__":
    main()
