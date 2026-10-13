# 9.16：宇宙方差基准与 Noise 恢复方差的 Fisher 预测

文档更新日期：2026-10-14；以下数值来自2026-10-13已完成并保存的运行结果。
当前模型按指定公式计算：

$$
\boxed{D_{\rm ana}=P_{\rm HI}^2,\qquad
D_{{\rm total},m}=P_{\rm HI}^2+P_{\rm HI}^2\widetilde A_m,\quad
m\in\{\mathrm{linear},\mathrm{spline}\}.}
$$

本次复用已有体素方差和 `noise_hkl_N200.h5`，只重新计算 Fisher。
解析项只保留宇宙方差；仪器噪声仍通过查询 Noise 时使用的 RMS 进入额外方差项。
本文及同目录结果均采用上述分母。

## 1. 计算流程与输入

日常入口为 [9.16 notebook](../9.16-interferometer-telescopes.ipynb)。
Notebook 调用 `forecast_916.run_fisher_only(cfg, SURVEYS)`：

1. 读取现有 `voxel_noise_<survey>.joblib`，沿 `z_mid` 线性插值方差，再开平方，取得查询 RMS。
2. 读取已有 Noise 标定表，在原始分箱归一化，再按红移、RMS 和二维波数插值。
3. 更新完整仪器域解析 Fisher；共同域只积分一次，同时得到解析、线性 Noise 与 spline Noise 三组矩阵。
4. 保存新基准、参数误差和插值对比；输入及计算代码匹配时，后续运行直接读取缓存。

本入口不调用 `voxel_variance` 或 `FiniteBoxVoxelIntegrator`，不重新生成模拟或体素方差。
`voxel_modes_916.py` 和 `diagnose_noise_916.py` 保留原作用，本次不运行其计算。

| 配置 | 当前取值 |
| --- | --- |
| 红移箱宽 | Δz=0.01，每箱2个 Gauss 节点 |
| 总波数上限 | k≤0.363 Mpc⁻¹，Fisher 与原体素表一致 |
| 模拟盒子、网格 | 3000 Mpc/h、512³；h_mock=0.6766 |
| 已有体素边长 | 8.660028 Mpc |
| 观测时间 | 100000 小时，沿用已有输入 |
| Noise 数据 | 完整200×200二维表，`noise_fisher_only=False`、`noise_wedge=True` |
| 参数 | lnA、Omega_m、omega_b、omega_cdm、n_s、w0、f_NL |

Noise 表位置：
`/mnt/xiaodong/luantch/LSS21cm_freeloadXD/PosiRestore/FisherMess/noise_hkl/data/noise_hkl_N200.h5`。
共享 `baofisher_k.py`、`experiments.py` 仍位于原 RadioFisher 目录。

各配置的实际仪器频段红移边界如下；共同域还会受 Noise 红移与 RMS 支持裁剪。

| 配置 | 仪器 z 下界 | 仪器 z 上界 |
| --- | ---: | ---: |
| ska_mid1_interferom | 0.352768 | 3.058303 |
| chime_icyl | 0.775508 | 2.551015 |
| hirax_interferom | 0.775508 | 2.551015 |
| tianlai_icyl | 0.495164 | 1.582556 |
| meerkat_b1_interferom | 0.399415 | 1.448976 |

## 2. Noise 的物理定义与插值

Noise 表固定每层红移的宇宙信号，改变200个热噪声实现，统计恢复功率的条件散布。
在每个原始标定分箱先计算：

$$
R_b=\frac{\mathrm{tf\_var}_b}{\mathrm{tf\_mean}_b^2},
\qquad A_b=\frac{N_{{\rm cal},b}}{2}R_b.
$$

`N_cal` 是 Fourier 分箱模式数，200是噪声实现数，二者不能混用。
读取器已完成模式数转换，之后插值 A；不再乘模式数、除样本数或另除观测模式数。
A 无量纲，体素温度方差的单位为 mK²，二者用途不同。

- 两种方案沿 `(z, log sigma)` 使用相同的 Delaunay 线性权重。
- 线性方案在 `(k_perp, k_parallel)` 上作双线性插值。
- spline 方案使用 `RectBivariateSpline(kx=3, ky=3, s=0)` 直接插值同一二维 A 面。
- `s=0` 表示穿过原始标定点，不消除标定值自身的统计散布。
- 表外不外推；负值或非有限样条值在该点回退到线性值，保留相同模式域。

Fisher 波数使用 Mpc⁻¹，除以 h_mock 后查询原表的 h/Mpc 坐标。
两种方案共用相同的归一化、红移权重、输入 RMS 和有效性判断。

## 3. Fisher 公式与有效域

代码保留明确的三项写法：

```python
denom_ana = p_hi**2
denom_noise = p_hi**2 * a_noise
denom_total = denom_ana + denom_noise
```

其中 P_HI 为 HI 温度功率谱，单位 mK² Mpc³；D 的单位为 mK⁴ Mpc⁶。
正 k_parallel 半轴上的 Fisher 公式为：

$$
F_{ij}^{(m)}=\int dz\,\frac{dV/dz}{4\pi^2}
\int_{\mathcal D(z),\,k_\parallel>0}
 k_\perp\,dk_\perp\,dk_\parallel\,
\frac{P_{,i}P_{,j}}{D_m},
\qquad P_{,i}=\frac{\partial P_{\rm HI}}{\partial\theta_i}.
$$

观测模式数已经由体积和积分测度计入；体素窗口不再单独乘入 Fisher。
参数误差为 $\sigma_i=\sqrt{(F^{-1})_{ii}}$，下表的 f_NL 误差均对其他六个参数边缘化。
CMB 组合沿用原7×7先验，使用 $F_{\rm HI}+F_{\rm CMB}$。

### 两个模式域

- **full**：现有仪器基线、FoV、径向模式掩码与总 k 上限的交集。
- **common**：full 再与 Noise 的红移、RMS、二维波数有效支持取交集。

Noise 分箱中心的物理波数范围为：

$$
0.002696108\le k_\perp\le0.511754277,\qquad
0.002320448\le |k_\parallel|\le0.361865957
\quad[\mathrm{Mpc}^{-1}].
$$

实际积分同时满足总 $k\le0.363\,\mathrm{Mpc}^{-1}$ 及仪器掩码。
原39个低 k 箱是筛选子集，不是完整表的有效边界；本次使用完整表。
无支持点从共同域解析、线性和样条三个对照同时排除，不外推、不填零误差。

### 新分母的直接物理含义

$$
\boxed{\frac{D_{\rm Noise}}{D_{\rm ana}}=\widetilde A,\qquad
\frac{\text{加入 Noise 后的单模信息}}{\text{解析单模信息}}
=\frac{1}{1+\widetilde A}.}
$$

因此 A=1 时，该模式的信息减半。只有 A 对所有相关模式近似恒定时，
才可用 $\sigma_{\rm total}/\sigma_{\rm ana}\simeq\sqrt{1+A}$ 估计整体误差。
实际结果还取决于模式权重和参数简并，应使用完整矩阵求逆。

## 4. 新分母运行结果

### 共同域、未加入 CMB 的 σ(f_NL)

下表三个结果分别采用以下分母，积分域、信号导数和红移权重保持一致：

$$
D_{\rm ana}=P_{\rm HI}^2,
$$

$$
D_{{\rm total},\mathrm{linear}}
=P_{\rm HI}^2+P_{\rm HI}^2\widetilde A_{\mathrm{linear}},
\qquad
D_{{\rm total},\mathrm{spline}}
=P_{\rm HI}^2+P_{\rm HI}^2\widetilde A_{\mathrm{spline}}.
$$

表中均为对其余六个参数边缘化后的 $\sigma(f_{\rm NL})$，未加入 CMB 先验。
样条结果包含前述负值或非有限值回退到线性的处理。
数值来自 `noise_interpolation_comparison.csv`，保留四位小数。

| 望远镜 | 解析 | 解析＋Noise 线性 | 解析＋Noise spline | Noise增幅 |
| --- | ---: | ---: | ---: | ---: |
| SKA-MID | 2.1097 | 2.9800 | 2.9792 | 41.252% |
| CHIME | 5.3741 | 7.2664 | 7.2646 | 35.213% |
| HIRAX | 4.8360 | 5.9957 | 5.9948 | 23.981% |
| Tianlai | 26.4907 | 29.1194 | 29.1178 | 9.923% |
| MeerKAT | 47.4653 | 66.9245 | 66.9063 | 40.997% |

“Noise增幅”为 $100(\sigma_{\rm linear}/\sigma_{\rm ana,common}-1)\%$，由未舍入数值计算；
“样条变化”为 $100(\sigma_{\rm spline}/\sigma_{\rm linear}-1)\%$。
**加入 Noise 后，误差增大约9.9%–41.3%；两种插值的差异小于0.028%。**
完整仪器域的解析结果（full）另见 `constraints_<survey>.csv` 中的 `sigma_ana_full`。

### 加入原 CMB 先验

| 配置 | common解析+CMB | 线性Noise+CMB | 样条Noise+CMB | Noise增幅 | 样条变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ska_mid1_interferom | 1.527639 | 2.060863 | 2.060404 | 34.905% | -0.022300% |
| chime_icyl | 3.084033 | 3.619203 | 3.618794 | 17.353% | -0.011292% |
| hirax_interferom | 2.869620 | 3.312395 | 3.312072 | 15.430% | -0.009753% |
| tianlai_icyl | 4.887300 | 4.913358 | 4.913343 | 0.533% | -0.000299% |
| meerkat_b1_interferom | 5.006416 | 5.043796 | 5.043773 | 0.747% | -0.000462% |

### 结果解释

加入 Noise 后，未加 CMB 的共同域 f_NL 误差增大约10%–41%。
线性与 spline 的差异仍很小，最大约0.0272%；加入原 CMB 先验后最大约0.0223%。
当前预测对这两种插值选择不敏感，光滑程度本身不能证明样条更准确。

full 与 common 的差异反映有效域裁剪，不能当作 Noise 引起的误差增加。
本次保持原信号导数和仪器掩码，未另行扩大到仪器直接覆盖之外的恢复模式。

回退占比按体积和 $k_\perp\,dk_\perp\,dk_\parallel$ 测度加权，不是 f_NL 信息占比：

| 配置 | 样条回退模式权重占比 |
| --- | ---: |
| ska_mid1_interferom | 0.0000000% |
| chime_icyl | 0.0000079% |
| hirax_interferom | 0.0001022% |
| tianlai_icyl | 0.0047132% |
| meerkat_b1_interferom | 0.0000000% |

## 5. 验证和物理适用范围

本次检查：

- notebook 在干净内核中完成，所有五个配置的参数误差均有限，无错误输出。
- 原体素 CSV/joblib、共享源码和配置共18个受保护文件的 SHA256 保持不变。
- 新基准与插值对比使用同次计算的共同域矩阵；序列化矩阵、CSV 和参数误差核对一致。
- 五个配置均满足：共同域解析信息不超过完整域解析信息，加入非负 Noise 不增加信息。
- 通过40,000个原生节点还原、域外拒绝和负样条回退检查。
- 在代表红移 HIRAX z=2.501，用对数功率谱导数独立重建三种 Fisher 分母；
  验证时禁止调用热噪声功率和体素方差计算。

独立解析矩阵的最大归一化差异为 2.34e-16；
求积加密后，三组参数误差的最大相对变化为 1.24e-05。
本次干净内核执行耗时 1701.3 秒；后续匹配缓存时直接读取。

数值表的末位差别用于同节点对比，不代表已验证相同位数的物理精度。
本次只重新执行 Fisher 相关验证；未变化的体素几何与读取器验证记录沿用已有结果。

物理适用范围：

- Noise 是固定宇宙信号下的条件散布；当前采用宇宙方差与该条件方差的对角叠加近似。
- 白噪声体素 RMS 匹配不等于完整的各向异性仪器噪声谱匹配。
- 模式数迁移采用近似1/N缩放，忽略箱间协方差；完整表之外不外推。
- 恢复后的 f_NL 响应仍用现有 HI 信号导数近似，尚需独立模拟标定。
- HIRAX 缺少基线文件，继续使用6–200 m均匀基线模型。

## 6. 输出与运行方式

| 文件 | 内容 |
| --- | --- |
| `voxel_noise_<survey>.csv/.joblib` | 复用的体素方差输入，未重新计算或重写 |
| `constraints_<survey>.csv` | full解析、common解析、线性Noise、样条Noise及各自CMB组合 |
| `fisher_<survey>.joblib` | 四组矩阵、参数顺序、新分母定义和复用体素文件的哈希 |
| `noise_interpolation_comparison.csv/.joblib` | 两种插值的 f_NL 对比、全部参数误差及缓存指纹 |
| `noise_coverage_<survey>.csv`及合表 | RMS、支持状态、模式覆盖和Noise/解析比值 |
| `noise_calibration_summary.json` | 各配置的覆盖与回退汇总 |
| `validation.json`、`execution_summary.json` | 数值验证、配置及保存结果一致性 |

体素文件保留其生成时的元数据；本次 Fisher 定义以 `fisher_<survey>.joblib` 为准。
体素合表仍为988行，843行在实际频段内；频段外的 NaN 表示没有观测数据。

日常只需运行 notebook。需要明确强制更新 Fisher 时可调用：

```python
results = run_fisher_only(cfg, SURVEYS, reuse_saved=False)
```

该调用仍然复用体素输入。相关验证命令：

```bash
/software/anaconda3/bin/python3 tests/verify_916.py --fisher-only
/software/anaconda3/bin/python3 tests/check_common_kmax.py
```

当前代码和结果统一在 `9.16/`，结果只保存在其 `outputs_916/` 子目录。
代码和运行结果已于2026-10-13同步至 GitHub 的 `9.16` 目录；
2026-10-14仅更新本文中的结果展示及公式说明，沿用已保存的数值。
