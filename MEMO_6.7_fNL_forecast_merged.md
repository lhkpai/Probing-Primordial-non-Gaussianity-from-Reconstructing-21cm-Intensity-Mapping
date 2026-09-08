# 合并备忘录：21 cm IM f_NL Fisher 预报 + 噪声口径辨析（6.7 / baofisher 全量）

> 整合日期：2026-09-09
> 由两份 memo 合并（逻辑去重、冲突已标注）：
> ① `MEMO_6.7_interferometer_fNL_forecast.md` —— 侧重「6.7 整合版 notebook + AI 协作演进 + 模块宇宙学」
> ② `memo_6.7_noise_fisher_baofisher.md` —— 侧重「噪声/体素方差口径三层辨析 + baofisher 版本关系 + arXiv 论文对比」
> 语言：中文（术语保留英文）。标注「⚠️推断」= 据代码注释/结构反推；「✅实证」= 本会话 diff/运行核实。

---

## 0. 版本地图（先读：避免混淆）

同一套物理代码有两个并行 notebook 版本，**两份 memo 分别对应其中一份**：

| 分支            | 文件                                                      | 特征                                                                                                                                                                                                                                | 对应 memo |
| --------------- | --------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- |
| **f_NL 整合版** | `research/6.7-interferometer-telescopes-2.ipynb`          | 唯一 code cell（~1390 行，末尾单行 `main()`）；顶部 `bf.KMIN_EFF=5e-3`、`bf.KMAX_EFF=0.35`；`SurveyConfig.k_min/k_max=5e-3/0.535`；mock 用 `fisher_sum_mock_zinterp`（**z 三维插值**, 越界 fill=0）；cell 内含 6 台望远镜 mock 对照 | ①         |
| **噪声对照版**  | `research/6.7-interferometer-telescopes.ipynb`（无 `-2`） | 4 cells：主 cell(exec) + 诊断 markdown + 诊断 cell + **严格单碟 vs 干涉对照 cell**（`T_OBS_H=1e4 h`）；顶部 `bf.KMIN_EFF=1e-4`、`k_min/k_max=1e-4/1.0`；mock 用旧 `fisher_sum_mock`（**2D 插值 + bin 序号硬配**, 越界 fill=inf）    | ②         |

库文件（两分支共用同一份）：

| 文件                                                               | 状态                                                                                     |
| ------------------------------------------------------------------ | ---------------------------------------------------------------------------------------- |
| `bao21cm-master/bao21cm-master/radiofisher/baofisher.py`           | = `baofisher-original.py`，**逐字节相同**（原版 Bull & Ferreira，未改动）                |
| `radiofisher/baofisher_k.py`                                       | **唯一被改动的库**：= baofisher + `KMIN_EFF`/`KMAX_EFF` 自定义各向同性 k 掩码（详见 §6） |
| `radiofisher/experiments.py`                                       | 望远镜/宇宙学/前景配置（本会话未修改）                                                   |
| `research/mock_fisher_arr.joblib`                                  | mock 阵列方差 `fisher_arr=Var(T)/⟨T⟩²`，6 个 z 层，每层 200×200 (k⊥, k∥)                 |
| `research/fisher_1.3.2.1_DirctTF_GenNoise_AssisHKL_Tutorial.ipynb` | mock 生成教程（100 实现 → 平均 → 导出 6 层 joblib）                                      |

---

## 1. 项目一句话定位与科学目标

用 **21 cm 中性氢（HI）强度映射（IM）** 的 Fisher 矩阵预测约束**局部原初非高斯参数 f_NL**：

1. **6 台/组望远镜 × 单碟（self-correlation）或干涉（interferometer）模式**横向对比；
2. 方差 = **解析高斯方差 + mock（模拟）方差**：
   $$\mathrm{Var}_{\rm tot}=\underbrace{\big(P_{\rm HI}+P_{\rm noise}\big)^{2}}_{\text{解析口径}}\;+\;\underbrace{f_{\rm arr}\cdot P_{\rm HI}^{2}}_{\text{mock},\;f_{\rm arr}={\rm Var}(T)/\langle T\rangle^{2}}$$
3. 信号/噪声/偏导全部在 $(k,\mu)$ 网格上按 z-bin 求和。

$f_{\rm NL}$ 进入信号的途径是**尺度相关偏压**：

$$
b_{\rm HI}(k,z)=b_{0}(z)+\underbrace{3(b_{0}-1)f_{\rm NL}\,\alpha(k,z)}_{\text{原初非高斯偏压}},\qquad
\alpha(k,z)=\Big(\frac{H_{0}}{c}\Big)^{2}\Omega_{m}\delta_{c}\,\frac{1}{T(k)k^{2}}\,\frac{1}{D(z)}
$$

- `baofisher` 提供「噪声引擎」（`Cnoise`/`Cfg`/响应函数）与望远镜参数；
- notebook 用 **classy** 算线性物质功率谱/增长，自行实现信号、偏导、Fisher 求和与输出；
- 转移函数 $1/(T(k)k^{2})$ 来自缓存 CAMB `transfer_bf_cache.dat`（`bf.deriv_transfer`，同步规范、$k_{\rm ref}=2\times10^{-3}h\,{\rm Mpc^{-1}}$ 归一，Jeong & Komatsu 2009 约定）。

---

## 2. 公式框架（两 memo 共用事实）

### 2.1 信号模型

$$P_{\rm HI}(k,\mu,z)=\bar T^{2}(z)\,b^{2}(k,z)\,P_{\rm m}(k,z)\,\big(1+\beta(k,z)\mu^{2}\big)^{2}\,e^{-\mu^{2}(k\,\sigma_{\rm nl})^{2}}$$

- $\bar T(z)$：平均亮温（mK），baofisher 同款二次拟合：$5.5919\times10^{-2}+2.3242\times10^{-1}z-2.4136\times10^{-2}z^{2}$；
- $b_{0}(z)=(b_{\rm HI0}/0.677105)[0.66655+0.17765z+0.050223z^{2}]$，$b_{\rm HI0}=0.677$；
- $\beta=f/b_{\rm HI}$：RSD 参数（Kaiser）；$\sigma_{\rm nl}=7\,{\rm Mpc}$：FoG 阻尼；
- $\delta_{c}=1.686$；$c=299792.458\,{\rm km/s}$。

### 2.2 参数集与偏导

- 7 参数：`[lnA, Omega_m, omega_b, omega_cdm, n_s, w0, f_NL]`；
- **f_NL 偏导解析**（RSD 因子中 β 的 f*NL 依赖在推导中成对抵消，仅保留尺度相关偏置项）：
  $$\frac{\partial P*{\rm HI}}{\partial f*{\rm NL}}=2\bar T^{2}b*{\rm tot}\,[3(b_{0}-1)\alpha]\,P*{m}(1+\beta\mu^{2})\,e^{-\mu^{2}k^{2}\sigma*{\rm nl}^{2}}$$
- 其余 6 参数**中心差分**（步长 $0.025\,|v|$），每步重建 classy；
- Planck CMB 先验：内嵌 7×7 矩阵，f_NL 对角 $3.8447\times10^{-2}$（≈σ(f_NL)=5.10），与 21cm 无交叉项。

### 2.3 Fisher 求和

- 解析：$F_{ab}=\sum_{\rm bins}\sum_{k}\sum_{\mu} w_{k,\mu}\,\dfrac{\partial_{a}P\,\partial_{b}P}{(P_{\rm HI}+P_{\rm noise})^{2}}$，$w_{k,\mu}=\dfrac{V_{\rm shell}k^{3}}{8\pi^{2}}\Delta\ln k\cdot\dfrac{2}{N_{\mu}}$（对应 $\int d^{3}k/(2\pi)^{3}$ 离散化）；$V_{\rm shell}=\tfrac{4\pi}{3}[r^{3}(z_{\rm hi})-r^{3}(z_{\rm lo})]f_{\rm sky}$；
- mock：分母换 $(P_{\rm HI}+P_{\rm noise})^{2}+f_{\rm arr}(z_{\rm mid};k_{\perp},k_{\parallel})P_{\rm HI}^{2}$（叠加细节见 §5）。

---

## 3. notebook 模块结构与物理含义（以整合版 `…-2.ipynb` 为准；对照版差异另注）

### 3.1 单 cell 结构地图（整合版，~1390 行）

| 区段   | 内容                                                                                                                                      |
| ------ | ----------------------------------------------------------------------------------------------------------------------------------------- |
| 头部   | matplotlib usetex=False；import classy/baofisher_k/experiments；`bf.KMIN_EFF=5e-3`+`bf.KMAX_EFF=0.35`；`C_LIGHT_KMS/DELTA_C/BF_INF_NOISE` |
| 1      | `normalize_mode`、`TelescopeSpec`、`TELESCOPES`（~70 条 `{望远镜}_{mode}`）                                                               |
| 2b     | mock 常量（`MOCK_FILE`、`MOCK_Z_LO/HI=0.5/3.0`、`MOCK_Z_CENTERS`）+ 给 6 台目标望远镜配 `mock_file`                                       |
| 2      | 通用工具 `fisher_submatrix/expt_to_f_sky/expt_to_zmin_zmax/bin_dnutot_MHz`                                                                |
| 3      | `SurveyConfig`（+`load()`）                                                                                                               |
| 4      | `CosmologyEngine`（classy 引擎 + P_noise 单点/网格向量化）                                                                                |
| 5      | `HIFisherForecast`（bin_data + `fisher_sum_analytic` + `fisher_sum_mock_zinterp`）                                                        |
| 6 / 6b | 输出层（绘图、鲁棒 y 轴）；realspace voxel 方差表+保存                                                                                    |
| 7–9    | Planck CMB Fisher、约束表、`run_forecast`、engine 缓存、`main()`                                                                          |

### 3.2 关键模块物理速览

- **`normalize_mode`**：`cylinder→icyl`（否则 `mode[0]!='i'` 使 baofisher 静默走单碟分支 → 物理错误）。
- **`TelescopeSpec(expt, extra, mock_file, mode_override)`**：`mock_file` 是"叠加 mock"的唯一开关；n(x) 来源：✅文件 → `load_interferom_file`；⚠️缺文件 → 降级 uniform（`Dmin/Dmax`）；🔢配置无 n(x) 且干涉 → 需 extra 补 Dmin/Dmax。
- **`expt_to_zmin_zmax`**：望远镜 z 窗口唯一来源 $z_{\min}=\nu_{\rm line}/\nu_{\rm max}-1.1028,\ z_{\max}=\nu_{\rm line}/(\nu_{\rm max}-\Delta\nu_{\rm tot})-0.8083$（1.1028/0.8083 为**自定义带宽边缘换算**；baofisher 内部用 $-1.0$，⚠️差异约为半 bin 补偿口径）。
- **`SurveyConfig`**：`t_obs_hours`（默认 1e5 h）为唯一时间入口；`load()` 装配 mode/extra/n(x) 并填 `f_sky`、`z_bins`。
- **`CosmologyEngine`**：Planck2018 风格 fiducial（$\ln A_s=\ln2.1\times10^{-9}$、$\Omega_m=0.3153$、$\omega_b=0.02237$、$\omega_{cdm}=0.1200$、$n_s=0.9649$、$w_0=-1$、$\sum m_\nu=0.06\,{\rm eV}$、$f_{\rm NL}=0$；$h=\sqrt{(\omega_b+\omega_{cdm})/\Omega_m}$）；classy 参数 `output=mPk`、`P_k_max_h/Mpc=5`、`halofit`、`z_max_pk=z_{\max}+0.5`。
  - 信号：`HI_power_spectrum`（§2.1）；噪声：`P_noise_grid` 把 (k,μ)→(q,y)（$q=r k\sqrt{1-\mu^{2}},\, y=r_{\nu}k|\mu|$，$r_{\nu}=(1+z)^{2}/H$）调 `bf.Cnoise`(±`Cfg`)，乘 $r^{2}r_{\nu}$ 转功率谱单位；凡噪声 ≥ $0.5\times{\rm INF\_NOISE}$ 判 `inf`（=模式弃用）。
- **`HIFisherForecast`**：`compute_bin_data` 逐 z-bin 产出 P_HI/P_noise/derivs；两个求和入口见 §2.3/§5。
- **输出层**：`robust_ylim`（有限值 log10 空间 [2%,98%] ±0.2 dex）；`build_curves` 取"中间 z + μ≈0.5"曲线，`P_noise≥1e4·max(P_HI)` 掩码；5 方差图 + 3 功率图 + 7 偏导图 + 约束表 CSV。
- **realspace voxel 方差（6b，独立口径）**：Gaussian voxel 窗 $W^{2}=e^{-B}$，$B=k^{2}[(1-\mu^{2})/\sigma_{k\perp}^{2}+\mu^{2}/\sigma_{k\parallel}^{2}]$，$\sigma_{k\parallel}=\sqrt{16\ln2}\,\nu_{\rm line}/(\Delta\nu\,r_{\nu})$，$\sigma_{k\perp}=\sqrt{16\ln2}/(r\,\theta_{\rm FWHM})$；$\sigma^{2}_{\rm sig}=\int\frac{d^{3}k}{(2\pi)^{3}}P_{\rm HI}W^{2}$；$\sigma^{2}_{\rm noise}=[{\rm noise\_rms\_per\_voxel}]^{2}$；输出每望远镜 CSV+joblib。

### 3.3 对照版（`…-telescopes.ipynb`）与本版差异

- k 窗口：`bf.KMIN_EFF=1e-4`、`SurveyConfig.k_max=1.0`（更宽的理想化窗口）；
- mock 求和仍为**2D 序号版**（`fisher_sum_mock`，`fill_value=np.inf`，要求 `len(mock_list)==nbins`）；
- 多出的**第 4 cell**：`ska_mid1` dish vs interferom 同参数严格对照（§4 的核心实验，`T_OBS_H=1e4 h`）。

---

## 4. 噪声与「体素方差」口径辨析（对照版核心；本会话最重要的物理结论）

### 4.1 ttot 的物理意义

- `ttot=观测小时×3600×1e6`，单位 **s·MHz**（1e4 h → 3.6e13 s·MHz）；`experiments.py` 默认 `SURVEY['ttot']=10e3*HRS_MHZ`（=1e4 h）。
- **物理图像**：整阵列所有天线**同时**对同一片天区积分，ttot 是共用的墙钟时间（×带宽）。碟数/基线多重性**不**进 ttot，而是经 `Ndish`（单碟）或 `n(u)`（干涉）进入噪声。
- $P_{\rm noise}\propto 1/t_{\rm tot}$ ⇒ 改 ttot 只缩放绝对尺度，**所有比值与 t 无关** —— 1e5→1e4 h 测试正是利用该判据（见 §4.6）。

### 4.2 口径 (A)：convenience 辅助公式（**本就在原始 baofisher**，非 baofisher_k 新增）

- 单碟 `noise_rms_per_voxel`：$\sigma_{d}^{2}=\dfrac{T_{\rm sys}^{2}S_{\rm area}}{\Delta\nu\,t_{\rm tot}\,\theta_{\rm FWHM}^{2}\,N_{\rm dish}N_{\rm beam}}$，$T_{\rm sys}=T_{\rm inst}+60{\rm e}3\big(\tfrac{300(1+z)}{\nu_{\rm line}}\big)^{2.55}\,{\rm mK}$
- 干涉 `noise_rms_per_voxel_interferom`：$\sigma_{i}^{2}=\dfrac{T_{\rm sys}^{2}S_{\rm area}\,{\rm FOV}^{2}}{\Delta\nu\,t_{\rm tot}}$，${\rm FOV}=(\lambda/D_{\rm dish})^{2}$

### 4.3 口径 (B)：严格逐模 Fisher 噪声（真正进入 F 的量）

$$P_{\rm noise}(k,\mu,z)=C_{\rm noise}(q,y)\cdot r^{2}\cdot r_{\nu},\qquad C_{\rm noise}=\frac{T_{\rm sys}^{2}V_{\rm survey}}{n_{\rm pol}\,t_{\rm tot}\,\Delta\nu_{\rm tot}}\times\text{(mode 响应)}$$

- 单碟响应：高斯束窗 $e^{B_{\rm tot}}$；干涉响应：$e^{B_{\parallel}}/n(u)$，$n(u)=n(x)/\nu^{2}$（$x=u/\nu$，来自 n(x) 文件或 uniform Dmin/Dmax）；
- 前景楔/非线性尺度等截止在「默认模式」生效，自定义 `KMIN/KMAX` 模式下被各向同性球形掩码取代（§6）。

### 4.4 W_vox（束窗，供 realspace voxel 口径）

- 单碟：$\sigma_{k\perp}\approx0.08\to0.009\,{\rm Mpc^{-1}}$（z=0.5→3），$\sigma_{k\parallel}$ 可忽略（频带极宽）；
- 干涉：真正 uv 平面 top-hat 型覆盖（+沿视线频窗）。

### 4.5 单碟 vs 干涉：三层对比（**关键结论**）

**(1) 口径 (A) 之比 —— 纯定义伪差，不可比。**
$$\frac{\sigma_{i}^{2}}{\sigma_{d}^{2}}=\underbrace{\theta^{6}N_{\rm dish}N_{\rm beam}}_{\sim10^{-8}\text{–}10^{-6}}$$
两式「体素」定义不同（单碟=一波束；干涉用 FOV² 归一），比值无物理意义（表中 aux_i/d≈1e-8 即此伪差）。

**(2) 同一合成体素 —— 干涉 ≈ 好 95×。**
噪声 $\propto T_{\rm sys}^{2}/(\Delta\nu\,t\,N_{\rm eff})$：单碟 $N_{\rm eff}=N_{\rm dish}=190$；干涉 $N_{\rm eff}\approx N(N-1)/2=17955$（独立基线数）⇒ ${\rm Var}_{i}/{\rm Var}_{d}\approx190/17955\approx1/95$（rms ≈1/9.7）。——「95×」来源。

**(3) 逐模 Fisher 口径 (B)（对 f_NL 唯一有意义）—— 结论相反且随 z 反转。**
见 §4.6：低 z 低 k 干涉单模更吵 10–50×（短基线缺失、无零间距自相关）；z≈1.3–1.6 交叉；高 z 单碟因束窗使固定 k 的有效体积坍缩、噪声爆炸（P_d@0.05→1e13），干涉平稳、median 反超。p90≈45–68 恒定表明高 z 分布**双峰**：多数模式干涉仍吵，但单碟在受束压制的模式上灾难性放大。

**一句话**：(A) 的 10⁷–10⁸ 是定义伪差；(同体素)95× 是样本数 17955/190 的真实增益；(逐模 Fisher) 低 z 低 k 干涉反而差 10–50×，高 z 反转 —— 只有 (B) 与 f_NL 约束相关。

### 4.6 数值结果：ska_mid1 dish vs interferom（对照版，T_OBS_H=1e4 h）

配置：SKA1MIDbase1（190×15 m、dnu=0.1 MHz、Sarea=25e3 deg²=7.6154 sr、Tsys=Tinst+Tsky），k∈[5e-3,0.535]、μ=0.5 列、k=0.05；有效模式 ~4890–4915/5000（两分支一致）。

| z   | σ_d²(A) | σ_i²(A) | aux_i/d | P_d@0.05 | P_i@0.05 | i/d@0.05 | med_i/d | p90     |
| --- | ------- | ------- | ------- | -------- | -------- | -------- | ------- | ------- |
| 0.5 | 2.4e-2  | 4.1e-10 | 1.7e-08 | 4.9e+01  | 2.5e+03  | 5.1e+01  | 5.5e+01 | 6.8e+01 |
| 1.0 | 1.7e-2  | 1.6e-09 | 9.5e-08 | 8.3e+02  | 1.2e+04  | 1.5e+01  | 1.9e+01 | 6.6e+01 |
| 1.5 | 1.4e-2  | 5.1e-09 | 3.6e-07 | 3.0e+04  | 2.9e+04  | 9.6e-01  | 2.3e+00 | 6.3e+01 |
| 2.0 | 1.4e-2  | 1.5e-08 | 1.1e-06 | 4.5e+06  | 5.8e+04  | 1.3e-02  | 6.1e-02 | 5.9e+01 |
| 2.5 | 1.4e-2  | 3.9e-08 | 2.7e-06 | 3.6e+09  | 1.3e+05  | 3.6e-05  | 4.3e-04 | 5.2e+01 |
| 3.0 | 1.6e-2  | 9.6e-08 | 6.1e-06 | 1.8e+13  | 2.5e+05  | 1.4e-08  | 8.3e-07 | 4.5e+01 |

早前 1e5 h 对照：绝对值 ×10（∝1/t），**比值逐位不变** → 验证 §4.1 的 t 无关性。

### 4.7 mock 方差相对解析的幅值（整合版实证）

- 解析 ${\rm Var_{ana}}=(P_{\rm HI}+P_{\rm noise})^{2}$；mock ${\rm Var_{mock}}=f_{\rm arr}P_{\rm HI}^{2}$；$r\equiv{\rm Var_{mock}}/{\rm Var_{ana}}=f_{\rm arr}[{\rm SNR}/(1+{\rm SNR})]^{2}\le f_{\rm arr}$。
- 实测 $f_{\rm arr}$：median≈1.2e-4、p90≈6.8e-4、max≈4.7e-2（全部 <0.05）⇒ 加 mock 后 $\Delta\sigma/\sigma\approx\langle r\rangle/2\sim10^{-5}$（实测 1e-6–1e-5）—— **「加 mock 几乎不变」是物理合理而非 bug**（噪声主导压低 r）。
- mock 网格事实：k⊥∈[0.003985,0.7564]、k∥∈[0.00343,0.5348]（总 k 0.0053–0.9264），6 层共用同一 (200,) 对数均匀网格。
- **9 vs 15（旧 notebook）之谜**：旧版 mock 越界 `fill_value=inf` ⇒ 低 k 模式判死（含 k<0.005 全部）；新版 `fill_value=0` ⇒ 退回纯解析。差异仅来自越界语义。SKA 网格仅 ~43.5% 模式落在 mock 矩形内 → 语义影响巨大。

---

## 5. Fisher 求和细节：bin、mock z 对齐与越界语义

### 5.1 bin 选取（✅实证）

① z 窗口来自各望远镜自身频率配置（§3.2 `expt_to_zmin_zmax`）；② 统一 `Δz=0.5` → `n_bins=round(跨度/0.5)`，`linspace` + 端点修正；③ 所有望远镜共用 k–μ 网格。6 台 z_mid 表见 §8。

### 5.2 mock 叠加与 z 越界语义

- **积分域永远是望远镜自己的 bin**；mock 只是"被查询对象"：
  - z_mid 落在 mock 层之间 → **层间线性插值**（整合版 `fisher_sum_mock_zinterp` 把 6 层堆成 3D，`RegularGridInterpolator((z,k⊥,k∥), fill=0)`）；
  - z_mid 越出 mock 范围 → `np.clip` 到边界层（用边界 f_arr **常数延伸**）；
  - mock 有而望远镜无的层 → **忽略**；
  - k 越界 → `fill_value=0` 退回纯解析（**z/k 越界语义不对称**：clip vs 置零）。
- 对照版（`fisher_sum_mock`）：2D 插值 + bin 序号硬配 + `fill=inf`——已废弃，仅保留在旧分支。

### 5.3 单次 `run_forecast` walkthrough（ska_mid1_interferom 例）

1. `SurveyConfig.load()`：装配 expt（mode='interferom'、n(x) 加载、f_sky、z_bins=(0.250,3.250)）；
2. `get_engine`（按 z 窗口缓存 classy）→ `compute_bin_data`（6 bin × 50 k × 100 μ：P_HI、P_noise、7 组 dP/dθ）；
3. `fisher_sum_analytic` → 7×7 F → +CMB → σ；
4. 配了 mock_file → joblib 读 6 层 → `fisher_sum_mock_zinterp` → σ(F_ana+mock(+CMB))；
5. 打印 `解析 z_mid=[0.5…3.0]` vs `mock z=[0.5…3.0]`（SKA 逐点重合 = 回归自检）；
6. main() 汇总约束表 CSV + 图。

---

## 6. k 上下限旋钮 KMIN_EFF / KMAX_EFF 与 baofisher_k diff

### 6.1 机制

模块级旋钮（默认 `0.0` = 关闭 = 原版行为）：`>0` 时用各向同性 $k=\sqrt{k_\parallel^{2}+k_\perp^{2}}$ 球形掩码**取代**对应物理截止，`≤0` 时保留 baofisher 原始物理截止（向后兼容）。notebook 运行时覆盖：

- 整合版：`bf.KMIN_EFF=5e-3; bf.KMAX_EFF=0.35`（即有效窗口 $[5\times10^{-3},0.35]$ Mpc⁻¹）；
- 对照版：`bf.KMIN_EFF=1e-4`（k_max 由 `SurveyConfig.k_max=1.0` 控制）。

### 6.2 baofisher_k 相对 baofisher 的真实差异（✅实证：`diff -u`）

**4 个 hunk，+54/−18 行**（3165→3203，净 +38）。按语义拆分 = **6 个改动点**：

| 语义点               | 位置/函数                                 | 内容                                                                                          |
| -------------------- | ----------------------------------------- | --------------------------------------------------------------------------------------------- | --- | -------------------------------- |
| ① 旋钮定义           | 模块顶部（`INF_NOISE` 后）                | 新增 `KMIN_EFF=0.0`、`KMAX_EFF=0.0` 两块注释+变量                                             |
| ② 文件越界填充       | `load_interferom_file`                    | 自定义模式下 n(x) 越界 `fill_value=(nx[0],nx[-1])` 边缘值；默认仍 `1/INF_NOISE`               |
| ③ uniform 低 u       | `interferometer_response`（uniform 分支） | 自定义模式（`KMIN>0 or KMAX>0`）关 `Dmin` 低 u 截止                                           |
| ④ uniform 高 u / FOV | 同函数                                    | `Dmax` 高 u 截止仅在 `KMAX_EFF≤0` 保留；FOV 截止在自定义模式关闭（cylinder 本就关闭）         |
| ⑤ Cnoise 低 k        | `Cnoise` 尾部低端                         | 自定义模式：前景楔（                                                                          | k∥  | <kfg）关掉，改 $k<K_{\min}$ 掩码 |
| ⑥ Cnoise 高 k        | `Cnoise` 尾部高端                         | 自定义模式：非线性尺度 $k_{\rm nl0}(1+z)^{2/(2+n_s)}$（k*nl0=0.14）关掉，改 $k>K*{\max}$ 掩码 |

> 说明：除这 6 点外 `baofisher_k` 与 `baofisher` 其余内容一致（SciPy shim 等兼容层两边相同）。`noise_rms_per_voxel_interferom` **不是新增**，原始 baofisher.py 已含。

---

## 7. 与论文 arXiv:2602.03313v2 噪声定义对比（Yu & Wang, JCAP 07(2026)091, SKA-Mid AA4）

| 量                   | 论文公式/参数                                                                                                                 | baofisher / 6.7 对应物                                                                           |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| 单 visibility 热噪声 | $\sigma_{S}=k_{B}T_{\rm sys}/(A_{e}\sqrt{\delta\nu\,\tau})$ [Jy]，η=0.81、D=15 m、τ=30 s、δν≈426.5 kHz、Tsys≈26 K             | 相同 $T_{\rm sys}^{2}/(\Delta\nu\,t)$ 噪声地板结构；ttot=τ×通道×带宽等价量                       |
| uv 单元噪声          | $\sigma_{V'_N}(u_i)=N(u_i)\sigma_{S}$，自然权重，单元求和归一                                                                 | `n(u)` 基线密度进 $1/n_u$；单元内 $\sqrt{N}$ 平均 ↔ $1/\sqrt{n(u)}$                              |
| 实空间噪声图         | $T_{N}(\hat x)={\rm FFT}[V'_N(u)/\sum_i N(u_i)]$（模拟端）                                                                    | 解析端直接 $P_{\rm noise}=C_{\rm noise}r^{2}r_{\nu}$（同噪声功率两种表示）                       |
| 场/覆盖              | 16×16 独立 32×32×512 场块，400 h/场                                                                                           | Sarea 与 ttot 进 Vsurvey/每模式噪声                                                              |
| 观测窗               | FOV≈λ/D≈0.028 rad；主瓣与频率平滑（PSF）**忽略**                                                                              | baofisher **计入**高斯束窗 $e^{B}$（单碟）/ $e^{B_\parallel}/n(u)$（干涉）→ 更保守               |
| 前景/模式截断        | 楔形 $k_{\parallel}\le k_{\perp}\sin\theta\tfrac{E(z)}{1+z}\int_{0}^{z}\tfrac{dz'}{E(z')}+b$（θ=π/2,b=0.1）+ 硬截 k<0.3 h/Mpc | 默认模式：前景楔+非线性尺度；自定义模式：各向同性 $k\in[k_{\min},k_{\max}]$（都删低 k 大尺度模） |
| 天线数               | AA4：164 天线/10 km                                                                                                           | SKA1MIDbase1：**190** 碟（不同参考设计，不可直接对数字）                                         |

要点：① 论文=模拟端噪声，baofisher=解析端，映射见表；② baofisher 多保留主瓣/频窗束效应（论文忽略 PSF），故论文高 k 更乐观；③ 截断方向性不同：论文沿楔删低 k 模，6.7 全方向一刀切（对低 k⊥ 楔外模，6.7 更乐观）。

---

## 8. 代码演化史（合并：沿革 + AI 会话阶段 + 双 notebook 分支）

### 8.1 沿革（⚠️据注释/结构推断 + ✅本会话实测）

1. **原始 baofisher**：Bull & Ferreira 全参数 Fisher（含 transfer_fn 的 f_NL、binned P(k)），依赖 CAMB；`experiments.py` 提供望远镜。
2. **自有管线（早期 notebook，「4.30.1 原始方法」）**：换 classy + 缓存转移函数，收敛到 7 参数集，f_NL 用解析偏导；噪声继续调 `baofisher.Cnoise`。
3. **加入 mock 方差**：`fisher_1.3.2.1_DirctTF_GenNoise_AssisHKL_Tutorial.ipynb` 由 100 次实现 → 平均 → 导出 6 层 `mock_fisher_arr.joblib`，再叠加进 Fisher。
4. **`6.7-interferometer.ipynb`（单/少台版）**：新增 realspace voxel 方差表（注释「保留自 6.7-interferometer.ipynb」）。
5. **`6.7-interferometer-telescopes.ipynb`（A3 重构，对照版）**：`TELESCOPES` 分发表、`SurveyConfig`、`CosmologyEngine`、`HIFisherForecast`、规范化输出、`main()`。✅本会话补充：诊断/严格对照 cell（dish vs interferom、T_OBS_H=1e4 h 落盘重跑）。
6. **`6.7-interferometer-telescopes-2.ipynb`（整合版）**：把 5 个 cell 程序化整合为 **1 个 code cell**，并引入 **z 插值 mock** 与 **KMAX_EFF 旋钮**。

### 8.2 AI 会话阶段（针对整合版）→ 演进要点

| 阶段                   | 问题                                          | 关键结论/落地                                                                                              |
| ---------------------- | --------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| A 概念澄清 (Q1–5)      | 解析+mock 如何约束 f_NL；插值方式；z 是否对齐 | Fisher 全流程；(k,μ)→(k⊥,k∥) 插值；**当时无 z 插值、SKA 对齐是巧合** → 提出 4 步 z 对齐方案                |
| B z 对齐实现 (Q6–7)    | 开始改代码；各 cell 做什么、为何重复          | B2 补丁（`fisher_sum_mock_zinterp`）；重复根因 = `__main__` 在 Jupyter 恒真 + 新旧 run_forecast 各跑一轮   |
| C 单 cell 整合 (Q8–10) | 能否整合                                      | 一次性脚本 7 处替换 → 1 code cell ~1390 行（删 B1b/空/探测/B2 cell），末尾单行 `main()`；ast/结构自检 PASS |
| D 语义深挖 (Q11–13)    | bin 选取；z 越界；举例                        | 频率配置→Δz=0.5→linspace；z 越界 clip、k 越界置零、mock 高 z 层闲置等逐台语义（§5）                        |
| E k 旋钮 (Q14–16)      | 上限只依赖参数；做 KMAX_EFF；按钮放 notebook  | 双层把关说明 → 在 baofisher_k 实现对称 `KMAX_EFF`（§6）→ notebook 顶部 `bf.KMAX_EFF=0.35`                  |

### 8.3 对照版的噪声诊断（另立线，已并入 §4）

第 4 cell 对照实验揭示：辅助公式伪差、同体素 95×、逐模 Fisher 随 z 反转三层结论；ttot 无关性判据（1e5→1e4 比值不变）。

---

## 9. 关键参数速查表（合并；标注版本差异）

| 类别   | 参数                                       | 整合版(-2)                  | 对照版(telescopes)     | 说明                                   |
| ------ | ------------------------------------------ | --------------------------- | ---------------------- | -------------------------------------- |
| k 窗口 | `bf.KMIN_EFF`                              | 5.0e-3 Mpc⁻¹                | 1.0e-4 Mpc⁻¹           | 各向同性 k 下限                        |
|        | `bf.KMAX_EFF`                              | 0.35 Mpc⁻¹                  | 0（默认）              | 各向同性 k 上限（0=关）                |
|        | `SurveyConfig.k_min/k_max`                 | 5e-3 / 0.535                | 1e-4 / 1.0             | 求和网格（建议整合版 k_max 对齐 0.35） |
|        | `nk / n_mu`                                | 50 / 100                    | 同                     | (k,μ) 采样                             |
| 红移   | `delta_z`                                  | 0.5                         | 同                     | bin 宽                                 |
|        | z 换算常数                                 | −1.1028 / −0.8083           | 同                     | 频率→z 窗口边缘                        |
| 观测   | `t_obs_hours`                              | 1e5 h（SurveyConfig）       | 1e4 h（第4 cell 已设） | 比值与 t 无关                          |
| 信号   | $\bar T_b(z)$、$b_0(z)$、$\sigma_{\rm NL}$ | 二次拟合；bHI0=0.677；7 Mpc | 同                     | 见 §2.1                                |
| 噪声   | `include_Cfg` / `BF_INF_NOISE`             | False / 1e200               | 同                     | 前景残差项 / 无穷哨兵                  |
| mock   | `MOCK_Z_LO/HI`                             | 0.5 / 3.0（6 层）           | 同                     | 假设 z 中心                            |
| f_NL   | fiducial / 解析步长                        | 0.0 / 0.1                   | 同                     | 解析偏导                               |
| 宇宙学 | fiducial                                   | Planck2018 风格             | 同                     | 见 §3.2                                |
| 望远镜 | `SURVEYS`                                  | 6 台                        | 主对照 SKA1MIDbase1    | 见下表                                 |

6 台望远镜（整合版 `SURVEYS`，均为干涉/cylinder；`z_bins` 由频率配置换算）：

| 望远镜                | mode           | numax / Δν (MHz) | z 窗口      | bin 数 | z_mid                         |
| --------------------- | -------------- | ---------------- | ----------- | ------ | ----------------------------- |
| ska_mid1_interferom   | interferom     | 1050 / 700       | 0.250–3.250 | 6      | 0.50,1.00,1.50,2.00,2.50,3.00 |
| chime_icyl            | cylinder(icyl) | 800 / 400        | 0.673–2.743 | 4      | 0.93,1.45,1.97,2.48           |
| hirax_interferom      | interferom     | 800 / 400        | 0.673–2.743 | 4      | 同 CHIME（文件缺失→uniform）  |
| tianlai_icyl          | cylinder(icyl) | 950 / 400        | 0.392–1.774 | 3      | 0.62,1.08,1.54                |
| cv_z0to3_interferom   | interferom     | 1200 / 850       | 0.081–3.250 | 6      | 0.34,0.87,1.40,1.93,2.46,2.99 |
| meerkat_b1_interferom | interferom     | 1015 / 435       | 0.297–1.641 | 3      | 0.52,0.97,1.42                |

---

## 10. 风险、待办与纠错记录（合并去重）

### 10.1 风险与待办

1. **口径 (A) 不可跨 mode 比较**（辅助公式体素定义不同，比值含 $\theta^{6}N_{\rm dish}N_{\rm beam}$ 伪差）。「谁更好」必须先定口径：约束 f_NL 只用口径 (B)。
2. **自定义 k 掩码是乐观假设**：KMIN/KMAX 模式关闭前景楔/Dmin/FOV/n(x) 范围/非线性尺度，默认低 k 大尺度模、高 k 小尺度模全按可用处理；真实前景清洗未必到 $k_{\min}=5\times10^{-3}$。
3. **⚠️疑似单位问题（待核实）**：notebook 多处 $r_{\nu}=(1+z)^{2}/H(z)$ 未乘光速 c（H 单位 km/s/Mpc ⇒ r*ν 非纯 Mpc）；baofisher 内部是 `cosmo['rnu']=C(1+z)²/H`。因 q,y 由同一 r*ν 反推、在自定义掩码下自洽，主要影响通道束窗 $\sigma_{k\parallel}$ 被放大 ~3e5 倍（形同关闭，沿视线略乐观）；单碟/干涉同受影响，比值结论稳健但绝对口径需修正。
4. **n(x) 文件口径**：SKA 系文件（190 碟 dec30）与论文 AA4（164 天线/10 km）非同一参考设计，跨论文对数字前需统一。
5. **mock z 中心假设 0.5–3.0**：来自列表长度推断而非 mock 元数据；不符则改 `MOCK_Z_LO/HI`。
6. `load_interferom_file` 边缘填充仅由 `KMIN_EFF` 触发：只设 `KMAX_EFF` 时文件型干涉高 u 仍受文件覆盖范围约束。
7. **编辑器-磁盘同步（过程教训）**：notebook 修改须经编辑器可见方式保存；磁盘级 JSON 手术会造成内核跑旧版（假成功/不一致）。对照版第 4 cell 磁盘与内核均为 `T_OBS_H=1e4` 已重跑一致。

### 10.2 纠错/澄清记录（避免重复绕路）

1. `noise_rms_per_voxel_interferom` **本就在原始 baofisher**，非 baofisher_k 新增；baofisher_k 全部改动仅 §6.2 的 6 点。
2. **ttot ≠ 各碟时间之和**：所有天线同时看同一片天区，多重性经 Ndish/n(u) 进入噪声。
3. 「干涉方差比单碟小 10⁷」(A) 与「干涉更吵 10–50×」(B) **不矛盾**——前者是定义伪差，后者是低 k/低 z 逐模真实行为。
4. matplotlib：系统缺 `type1cm.sty` ⇒ `text.usetex=False`；DejaVu 无 CJK ⇒ 图注用英文。
5. 观测时间 1e5→1e4 h 只验证绝对尺度（∝1/t），比值不变——作为测试判据达成。
6. 整合版曾误述的「fill 边缘值触发条件」「噪声辅助公式是否新增」等已在 §0/§6 以实证 diff 更正。
