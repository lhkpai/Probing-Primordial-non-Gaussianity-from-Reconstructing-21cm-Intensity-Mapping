# 9.16：k 上限与径向上限改变后的 σ(f_NL) 对比

整理日期：2026-10-14。本文汇总三组已保存的七参数 Fisher 结果，不重新计算 Fisher 或体素方差。

**主要结论：** 总 $k$ 上限由 0.363 降至 0.2 后，解析含热噪声的误差增大约 **3.6%–60.0%**；
再加入 $k_\parallel\le0.01$ 后，误差变为此前的 **2.26–6.87 倍**。
解析不含热噪声时，两步变化分别为 **1.56–1.82 倍**、**2.57–6.07 倍**。
这些倍数均指 common 域、未加 CMB 的边缘化 $\sigma(f_{\rm NL})$。

## 1. 三组运行配置及公式

| 编号 | Fisher 总 k 上限 / Mpc⁻¹ | 独立径向上限 / Mpc⁻¹ | 改动 |
| --- | ---: | ---: | ---: |
| A | 0.363 | 未设置 | 较早的双分母基准 |
| B | 0.200 | 未设置 | 仅缩小总 k 上限 |
| C（当前） | 0.200 | 0.010 | 在 B 基础上缩小径向范围 |

“未设置独立径向上限”时，$k_\parallel$ 仍受总波数球面限制。令 $p=k_\perp$，正径向半轴的上边界为：

$$
\begin{aligned}
q_{\max,A}(p)&=\sqrt{0.363^2-p^2},\\
q_{\max,B}(p)&=\sqrt{0.2^2-p^2},\\
q_{\max,C}(p)&=\min\!\left(0.01,\sqrt{0.2^2-p^2}\right),
\qquad q=k_\parallel>0.
\end{aligned}
$$

上述波数均以 $\mathrm{Mpc}^{-1}$ 为单位。这里只写上边界；仪器基线、FoV、原有径向下限及有效模式掩码仍然保留。
$k=\sqrt{k_\perp^2+k_\parallel^2}$，因此 $k_\parallel\le0.01$ **不等于**总 $k\le0.01$。

两种解析分母与各自加入 Noise 后的总分母为：

$$
\begin{aligned}
D_{\rm ana}^{(N)}&=(P_{\rm HI}+P_N)^2,
& D_{{\rm total},m}^{(N)}&=(P_{\rm HI}+P_N)^2+P_{\rm HI}^2\widetilde A_m,\\
D_{\rm ana}^{(0)}&=P_{\rm HI}^2,
& D_{{\rm total},m}^{(0)}&=P_{\rm HI}^2+P_{\rm HI}^2\widetilde A_m,
\qquad m\in\{\mathrm{linear},\mathrm{spline}\}.
\end{aligned}
$$

$P_{\rm HI}$ 为 HI 温度功率谱，$P_N$ 为仪器热噪声功率谱；
$\widetilde A_m$ 为完整 $200\times200$ Noise 表经模式数归一化转换、插值后的无量纲额外恢复方差。

**$\widetilde A_m$ 的来源：先在原始分箱上转换，再插值。** 令 $b$ 表示 Noise 表的原始二维波数分箱：

$$
\begin{aligned}
R_b&=\frac{\mathrm{tf\_var}_b}{\mathrm{tf\_mean}_b^2},
& A_b&=\frac{N_{{\rm cal},b}}{2}\,R_b,\\
\widetilde A_m(\mathbf k,z)
&=\mathcal I_m[A_b]\!\left(
z,\sigma_a(z),\frac{k_\perp}{h_{\rm mock}},
\frac{\lvert k_\parallel\rvert}{h_{\rm mock}}\right),\\
D_{{\rm Noise},m}&=P_{\rm HI}^2\widetilde A_m.
\end{aligned}
$$

- `tf_mean`、`tf_var` 是固定宇宙场、改变热噪声实现得到的转移函数均值和方差；$R_b$ 是其无量纲相对方差。
- $N_{{\rm cal},b}$ 是标定模拟在该分箱内的 Fourier 模式数；$200\times200$ 指波数分箱网格，不是这里的模式数。
  $N_{\rm cal}/2$ 沿用 Gaussian 功率谱方差 $\mathrm{Var}(\widehat P_b)\simeq2D_b/N_{{\rm cal},b}$ 的模式计数约定，
  在对角协方差及 $1/N$ 缩放近似下，把箱平均相对方差转换为 Fisher 分母使用的系数。
- $\sigma_a(z)$ 为望远镜 $a$ 已有体素热噪声方差沿红移插值后开平方得到的温度 RMS。
  $\mathcal I_m$ 在 $(z,\log\sigma)$ 上共用线性插值，在二维波数上分别采用双线性或三次样条。
  波数除以 $h_{\rm mock}=0.6766$，将 Fisher 的 $\mathrm{Mpc}^{-1}$ 转为 Noise 表的 $h/\mathrm{Mpc}$。

对应代码为 `NoiseHKLTable._plane` → `NoiseCalibration.evaluate_amplitude`。
Fisher 直接使用 $P_{\rm HI}^2\widetilde A_m$；不再重复乘标定模式数，也不额外除以观测模式数。

“解析不含热噪声”仅说明解析分母没有 $P_N$，该模型的额外 Noise 项仍保留模拟中的热噪声恢复散布。

代码在正 $k_\parallel$ 半轴积分：

$$
F_{ij}=\int dz\,\frac{dV/dz}{4\pi^2}
\int_{\mathcal D(z),\,k_\parallel>0}
 k_\perp\,dk_\perp\,dk_\parallel\,
\frac{\partial_{\theta_i}P_{\rm HI}\,\partial_{\theta_j}P_{\rm HI}}{D},
\qquad
\sigma(f_{\rm NL})=\sqrt{(F^{-1})_{f_{\rm NL}f_{\rm NL}}}.
$$

各表的公式备注中，$F_{\rm common}[D]$、$F_{\rm full}[D]$ 分别表示在对应模式域、以 $D$ 为分母的 HI Fisher 矩阵，
$f$ 简记 $f_{\rm NL}$。A/B/C 分别使用本节列出的三组波数上限；$m$ 表示 linear 或 spline。

本文所有误差均对其余六个参数边缘化。参数顺序为
$\{\ln A,\Omega_m,\omega_b,\omega_{\rm cdm},n_s,w_0,f_{\rm NL}\}$；$f_{\rm NL}$ 及其误差无量纲。

**积分域的区别：**

- **full**：仪器有效模式域与本次总 $k$、径向上限的交集。
- **common**：full 再与 Noise 的红移、RMS 和二维波数有效支持取交集。
- 同一组运行内，两种解析分母及线性／spline 对照使用相同 common 域；A、B、C 的 common 域随上限变化而缩小。

三组共同设置为 $\Delta z=0.01$、观测时间 100000 小时、完整 Noise 表、
`noise_fisher_only=False`、`noise_wedge=True`、横向／径向积分设置 96／64、每个红移箱 2 个积分节点。
各望远镜红移边界与配置保持不变。体素标定均沿用总 $k$ 上限 0.363、
盒长 $3000\,h^{-1}\mathrm{Mpc}$、$512^3$ 网格及 $h=0.6766$ 的已有文件；改变 Fisher 截断不重新计算图像 RMS。

### 1.1 当前 C 配置的三个波数范围：z=1

下表为当前 $k_{\max}=0.2$、$k_{\parallel,\max}=0.01$ 配置在 **$z=1$、full 仪器域**中的有效范围包络。
全部为共动波数，单位均为 $\mathrm{Mpc}^{-1}$；已计入基线密度的有效支持和原有 FoV 截断。

| 望远镜 | $\lvert k_\parallel\rvert$ 范围 | $k_\perp$ 有效范围包络 | 总 $k$ 范围 |
| --- | ---: | ---: | ---: |
| SKA-MID | 0.001282–0.010000 | 0.065472–0.199996 | 0.065484–0.200000 |
| CHIME | 0.002243–0.010000 | 0.085501–0.199987 | 0.085531–0.200000 |
| HIRAX | 0.002243–0.010000 | 0.026189–0.199987 | 0.026285–0.200000 |
| Tianlai | 0.002243–0.010000 | 0.062649–0.199987 | 0.062689–0.200000 |
| MeerKAT | 0.002063–0.010000 | 0.058925–0.199989 | 0.058961–0.200000 |

这些上下限**不能任意组合成一个矩形积分域**；每个模式还必须同时满足：

$$
\boxed{
k_\perp^2+k_\parallel^2\le0.2^2,\qquad
\lvert k_\parallel\rvert\le0.01,\qquad n(u)>0,
\quad u=\frac{k_\perp r(z)}{2\pi}.
}
$$

径向下限和由此得到的横向上限为：

$$
\begin{aligned}
q_{\min}(z)&=\frac{2\pi\nu_{\rm line}}
{\Delta\nu_{\rm survey}\,r_\nu(z)}\,k_{\rm fg,fac},\\
p_{\max}(z)&=\min\!\left[p_{\rm inst,max}(z),
\sqrt{0.2^2-q_{\min}^2(z)}\right],\\
k_{\min}(z)&=\sqrt{[p_{\min}^{+}(z)]^2+q_{\min}^2(z)}.
\end{aligned}
$$

其中 $p=k_\perp$、$q=\lvert k_\parallel\rvert$；$p_{\min}^{+}$ 是同时满足基线正密度及 FoV 条件的横向下边界，
$p_{\rm inst,max}$ 是仪器基线支持的横向上边界。表中边界保留六位小数；零密度边界点本身不参与积分。
CHIME/Tianlai 的低端有零密度区，因此不能直接把基线文件的第一个横坐标当作有效下限。
HIRAX 仍采用 $D_{\min}=6$ m、$D_{\max}=200$ m 的均匀基线假设。

**适用范围：**

- 上下限随红移变化；本表只对应 $z=1$，不是各望远镜所有红移共用的边界。
- common 域还须与 Noise 标定支持取交集。在红移和输入 RMS 有效时，$z=1$ 的 common 径向下限约为
  $0.00232045\,\mathrm{Mpc}^{-1}$；本表列出的 full 下限不直接替代 common 下限。
- 功率比图固定 $k_\parallel=0.005\,\mathrm{Mpc}^{-1}$，其最低总波数为
  $\sqrt{[p_{\min}^{+}]^2+0.005^2}$；离散绘图采样的首个有效点还会略高于这个边界。
  因此图中的起点与本表总 $k$ 下限不完全相同。

## 2. 主要对比：解析 σ(f_NL)，common 域、未加 CMB

表中“B/A”“C/B”均为新误差除以旧误差，越大表示约束变弱越多。
σ 保留四位小数，倍数和百分比由未舍入数据计算。

### 2.1 解析含热噪声：D_ana=(P_HI+P_N)²

**公式备注：**

$$
D_{\rm ana}=(P_{\rm HI}+P_N)^2,\qquad
\sigma_{\rm ana}=\sqrt{\left(F_{\rm common}[D_{\rm ana}]^{-1}\right)_{ff}}.
$$

| 望远镜 | A：k≤0.363 | B：k≤0.2 | C：再限 k∥≤0.01 | B/A | C/B |
| --- | ---: | ---: | ---: | ---: | ---: |
| SKA-MID | 40.2700 | 41.7090 | 94.1607 | 1.0357 | 2.2576 |
| CHIME | 10.1875 | 11.7751 | 65.4691 | 1.1558 | 5.5600 |
| HIRAX | 9.1662 | 9.9560 | 30.9791 | 1.0862 | 3.1116 |
| Tianlai | 30.4783 | 48.7768 | 288.5516 | 1.6004 | 5.9158 |
| MeerKAT | 442.5503 | 508.9351 | 3495.4868 | 1.1500 | 6.8682 |

### 2.2 解析不含热噪声：D_ana=P_HI²

**公式备注：**

$$
D_{\rm ana}=P_{\rm HI}^2,\qquad
\sigma_{\rm ana}=\sqrt{\left(F_{\rm common}[D_{\rm ana}]^{-1}\right)_{ff}}.
$$

| 望远镜 | A：k≤0.363 | B：k≤0.2 | C：再限 k∥≤0.01 | B/A | C/B |
| --- | ---: | ---: | ---: | ---: | ---: |
| SKA-MID | 2.1097 | 3.8419 | 9.8684 | 1.8211 | 2.5686 |
| CHIME | 5.3741 | 8.9408 | 47.0023 | 1.6637 | 5.2571 |
| HIRAX | 4.8360 | 7.5651 | 24.6865 | 1.5643 | 3.2632 |
| Tianlai | 26.4907 | 47.5733 | 278.9654 | 1.7959 | 5.8639 |
| MeerKAT | 47.4653 | 86.6176 | 525.4527 | 1.8249 | 6.0663 |

## 3. 加入额外 Noise：common 域、未加 CMB

“解析”仍使用本节对应的解析分母；“＋线性”“＋spline”使用第1节对应总分母。
Noise 增幅定义为 $100(\sigma_{\rm linear}/\sigma_{\rm ana,common}-1)\%$。
为显示两种插值的小差别，本节 σ 保留五位小数。

### 3.1 解析含热噪声

**公式备注：** 三个结果列依次对应 $j=\mathrm{ana},\mathrm{linear},\mathrm{spline}$。

$$
\begin{aligned}
D_{\rm ana}&=(P_{\rm HI}+P_N)^2,\\
D_m&=(P_{\rm HI}+P_N)^2+P_{\rm HI}^2\widetilde A_m,\\
\sigma_j&=\sqrt{\left(F_{\rm common}[D_j]^{-1}\right)_{ff}}.
\end{aligned}
$$

| 运行 · 望远镜 | 解析 | ＋Noise 线性 | ＋Noise spline | Noise 增幅 |
| --- | ---: | ---: | ---: | ---: |
| A · SKA-MID | 40.27004 | 40.89607 | 40.89639 | 1.555% |
| A · CHIME | 10.18754 | 12.40669 | 12.40471 | 21.783% |
| A · HIRAX | 9.16623 | 9.93284 | 9.93256 | 8.363% |
| A · Tianlai | 30.47831 | 32.66125 | 32.66008 | 7.162% |
| A · MeerKAT | 442.55029 | 451.01616 | 451.01395 | 1.913% |
| B · SKA-MID | 41.70904 | 42.38516 | 42.38552 | 1.621% |
| B · CHIME | 11.77509 | 14.54957 | 14.54685 | 23.562% |
| B · HIRAX | 9.95597 | 10.85668 | 10.85629 | 9.047% |
| B · Tianlai | 48.77678 | 51.62349 | 51.62214 | 5.836% |
| B · MeerKAT | 508.93509 | 520.27442 | 520.27107 | 2.228% |
| C · SKA-MID | 94.16075 | 95.57088 | 95.57378 | 1.498% |
| C · CHIME | 65.46913 | 83.16543 | 83.12326 | 27.030% |
| C · HIRAX | 30.97914 | 33.37458 | 33.37346 | 7.732% |
| C · Tianlai | 288.55158 | 304.88474 | 304.87032 | 5.660% |
| C · MeerKAT | 3495.48681 | 3567.01858 | 3566.95408 | 2.046% |

### 3.2 解析不含热噪声

**公式备注：** 三个结果列依次对应 $j=\mathrm{ana},\mathrm{linear},\mathrm{spline}$。

$$
\begin{aligned}
D_{\rm ana}&=P_{\rm HI}^2,\\
D_m&=P_{\rm HI}^2+P_{\rm HI}^2\widetilde A_m,\\
\sigma_j&=\sqrt{\left(F_{\rm common}[D_j]^{-1}\right)_{ff}}.
\end{aligned}
$$

| 运行 · 望远镜 | 解析 | ＋Noise 线性 | ＋Noise spline | Noise 增幅 |
| --- | ---: | ---: | ---: | ---: |
| A · SKA-MID | 2.10967 | 2.97995 | 2.97917 | 41.252% |
| A · CHIME | 5.37405 | 7.26641 | 7.26456 | 35.213% |
| A · HIRAX | 4.83597 | 5.99568 | 5.99483 | 23.981% |
| A · Tianlai | 26.49067 | 29.11943 | 29.11776 | 9.923% |
| A · MeerKAT | 47.46533 | 66.92449 | 66.90633 | 40.997% |
| B · SKA-MID | 3.84190 | 5.42972 | 5.42838 | 41.329% |
| B · CHIME | 8.94076 | 11.91226 | 11.90898 | 33.235% |
| B · HIRAX | 7.56514 | 8.80789 | 8.80694 | 16.427% |
| B · Tianlai | 47.57333 | 50.60157 | 50.60002 | 6.365% |
| B · MeerKAT | 86.61763 | 122.15535 | 122.12075 | 41.028% |
| C · SKA-MID | 9.86841 | 13.95098 | 13.95027 | 41.370% |
| C · CHIME | 47.00230 | 63.62484 | 63.59242 | 35.365% |
| C · HIRAX | 24.68654 | 28.09193 | 28.09169 | 13.795% |
| C · Tianlai | 278.96541 | 296.84637 | 296.83010 | 6.410% |
| C · MeerKAT | 525.45275 | 741.77041 | 741.45295 | 41.168% |

### 3.3 插值差异汇总

定义 $\delta_{\rm spline}=100|\sigma_{\rm spline}/\sigma_{\rm linear}-1|\%$。
下表给出各组五个望远镜中的最大值，均未加 CMB。

| 运行 | 解析含热噪声 | 解析不含热噪声 |
| --- | ---: | ---: |
| A | 0.0160% | 0.0271% |
| B | 0.0187% | 0.0283% |
| C | 0.0507% | 0.0509% |

## 4. 加入相同 CMB 先验后的对比

使用 $F_{\rm joint}=F+F_{\rm CMB}$，再对七参数联合边缘化。
以下仍是 common 域，σ 保留五位小数。

### 4.1 解析含热噪声＋CMB

**公式备注：** $j=\mathrm{ana},\mathrm{linear},\mathrm{spline}$，在对应 HI Fisher 上加同一 CMB 先验。

$$
\begin{aligned}
D_{\rm ana}&=(P_{\rm HI}+P_N)^2,\\
D_m&=(P_{\rm HI}+P_N)^2+P_{\rm HI}^2\widetilde A_m,\\
\sigma_{j+\mathrm{CMB}}&=\sqrt{\left[\left(F_{\rm common}[D_j]+F_{\rm CMB}\right)^{-1}\right]_{ff}}.
\end{aligned}
$$

| 运行 · 望远镜 | 解析＋CMB | Noise 线性＋CMB | Noise spline＋CMB |
| --- | ---: | ---: | ---: |
| A · SKA-MID | 5.03526 | 5.03666 | 5.03666 |
| A · CHIME | 4.26019 | 4.45422 | 4.45410 |
| A · HIRAX | 4.06941 | 4.18694 | 4.18690 |
| A · Tianlai | 4.93468 | 4.95060 | 4.95060 |
| A · MeerKAT | 5.09928 | 5.09930 | 5.09930 |
| B · SKA-MID | 5.03647 | 5.03787 | 5.03787 |
| B · CHIME | 4.34053 | 4.53045 | 4.53033 |
| B · HIRAX | 4.19629 | 4.31250 | 4.31246 |
| B · Tianlai | 5.00746 | 5.01828 | 5.01828 |
| B · MeerKAT | 5.09939 | 5.09940 | 5.09940 |
| C · SKA-MID | 5.07999 | 5.08046 | 5.08046 |
| C · CHIME | 4.95821 | 5.00079 | 5.00074 |
| C · HIRAX | 4.85047 | 4.88485 | 4.88487 |
| C · Tianlai | 5.09018 | 5.09131 | 5.09131 |
| C · MeerKAT | 5.09987 | 5.09988 | 5.09988 |

### 4.2 解析不含热噪声＋CMB

**公式备注：** $j=\mathrm{ana},\mathrm{linear},\mathrm{spline}$，在对应 HI Fisher 上加同一 CMB 先验。

$$
\begin{aligned}
D_{\rm ana}&=P_{\rm HI}^2,\\
D_m&=P_{\rm HI}^2+P_{\rm HI}^2\widetilde A_m,\\
\sigma_{j+\mathrm{CMB}}&=\sqrt{\left[\left(F_{\rm common}[D_j]+F_{\rm CMB}\right)^{-1}\right]_{ff}}.
\end{aligned}
$$

| 运行 · 望远镜 | 解析＋CMB | Noise 线性＋CMB | Noise spline＋CMB |
| --- | ---: | ---: | ---: |
| A · SKA-MID | 1.52764 | 2.06086 | 2.06040 |
| A · CHIME | 3.08403 | 3.61920 | 3.61879 |
| A · HIRAX | 2.86962 | 3.31239 | 3.31207 |
| A · Tianlai | 4.88730 | 4.91336 | 4.91334 |
| A · MeerKAT | 5.00642 | 5.04380 | 5.04377 |
| B · SKA-MID | 2.64402 | 3.30574 | 3.30525 |
| B · CHIME | 3.75390 | 4.17545 | 4.17514 |
| B · HIRAX | 3.62792 | 3.94524 | 3.94504 |
| B · Tianlai | 4.99996 | 5.01309 | 5.01308 |
| B · MeerKAT | 5.06582 | 5.08238 | 5.08237 |
| C · SKA-MID | 4.32031 | 4.64211 | 4.64202 |
| C · CHIME | 4.84508 | 4.94223 | 4.94214 |
| C · HIRAX | 4.71833 | 4.80572 | 4.80572 |
| C · Tianlai | 5.08929 | 5.09070 | 5.09070 |
| C · MeerKAT | 5.09653 | 5.09822 | 5.09822 |

原代码的 CMB 先验直接包含 $F_{{\rm CMB},f_{\rm NL}f_{\rm NL}}=0.038447$，
且 $f_{\rm NL}$ 与其他参数的先验交叉项为零。

因此仅该先验给出 $\sigma_{\rm CMB}(f_{\rm NL})=1/\sqrt{0.038447}=5.09998$。
HI 信息很弱时，联合结果趋近这一数值；接近 5.10 主要由既有先验决定。

## 5. 补充：full 域的纯解析结果，未加 CMB

这里不要求 Noise 表支持，因此通常比第2节 common 解析结果更强。
本表对应输出列 `sigma_ana_full`；第2节对应 `sigma_ana_common`。
评估额外 Noise 的影响时，应使用同一 common 域内的对照。

### 5.1 解析含热噪声

**公式备注：**

$$
D_{\rm ana}=(P_{\rm HI}+P_N)^2,\qquad
\sigma_{\rm ana,full}=\sqrt{\left(F_{\rm full}[D_{\rm ana}]^{-1}\right)_{ff}}.
$$

| 望远镜 | A：k≤0.363 | B：k≤0.2 | C：再限 k∥≤0.01 |
| --- | ---: | ---: | ---: |
| SKA-MID | 38.1241 | 39.4679 | 83.4866 |
| CHIME | 10.0875 | 11.6506 | 62.8435 |
| HIRAX | 8.7947 | 9.5565 | 27.7378 |
| Tianlai | 30.1180 | 48.3736 | 283.2184 |
| MeerKAT | 441.1887 | 507.5362 | 3425.5259 |

### 5.2 解析不含热噪声

**公式备注：**

$$
D_{\rm ana}=P_{\rm HI}^2,\qquad
\sigma_{\rm ana,full}=\sqrt{\left(F_{\rm full}[D_{\rm ana}]^{-1}\right)_{ff}}.
$$

| 望远镜 | A：k≤0.363 | B：k≤0.2 | C：再限 k∥≤0.01 |
| --- | ---: | ---: | ---: |
| SKA-MID | 1.9534 | 3.5418 | 8.5452 |
| CHIME | 5.3206 | 8.8394 | 44.8595 |
| HIRAX | 4.6377 | 7.2528 | 21.8189 |
| Tianlai | 26.2048 | 47.1852 | 273.8175 |
| MeerKAT | 47.3176 | 86.3503 | 515.3781 |

## 6. 如何理解这些变化

1. **限制径向模式的影响更明显。** 本次 B→C 的误差增长倍数均大于 A→B；C 同时丢失大量径向模式和角向信息。
   对应二维积分域是一段很薄的 $k_\parallel$ 区间，而不是一个总 $k\le0.01$ 的小球。
2. **f_NL 对低 k 敏感，不代表其他模式可以忽略。** 高 k 与不同视线夹角的模式还能约束振幅、谱倾斜、增长等参数，
   缩小这些模式后，参数简并增强，边缘化的 $\sigma(f_{\rm NL})$ 可能明显增大。
3. **两种解析分母是不同协方差假设。** 含热噪声时，额外恢复误差相对解析项的比例为
   $R_N=[P_{\rm HI}/(P_{\rm HI}+P_N)]^2\widetilde A$；不含热噪声时为 $R_0=\widetilde A$。
   因此相同 Noise 对含热噪声基准的相对影响往往更小。
4. **本次插值选择的影响很小。** 三组运行中，未加 CMB 的线性／spline 差异均小于 0.051%，
   远小于改变模式范围或解析协方差造成的差别。
5. **物理解释仍沿用现有 Noise 模型。** Noise 作为额外条件恢复方差加入；恢复后的参数响应使用原信号导数近似，
   协方差忽略不同分箱之间的相关性。较小的“不含热噪声”误差表示更理想化的解析基准。

## 7. 数据来源与核对记录

- **A**：GitHub 固定提交 `aed0009`（2026-10-14 01:25，UTC+8）中的
  [双分母结果 CSV](https://github.com/lhkpai/Probing-Primordial-non-Gaussianity-from-Reconstructing-21cm-Intensity-Mapping/blob/aed000946d20fdd25f9e82bb8b1bf5bf29ab5b6c/9.16/outputs_916/fisher_denominator_comparison.csv)
  及同目录 joblib；另与原“不含热噪声”保存快照核对一致。
- **B**：2026-10-14 14:22、加入径向上限前保留的计算快照
  `/tmp/916_before_kpar01.joblib`。本次比较所用的 f_NL 数值已列入本文；
  纯解析 common 值另与 [implementation_notes.md 第1.4节](https://github.com/lhkpai/Probing-Primordial-non-Gaussianity-from-Reconstructing-21cm-Intensity-Mapping/blob/a053e4ee5e1aa623e0695f46445bc9334bab2245/9.16/outputs_916/implementation_notes.md#14-径向上限改变前后的解析约束) 一致。
- **C**：当前 [fisher_denominator_comparison.csv](https://github.com/lhkpai/Probing-Primordial-non-Gaussianity-from-Reconstructing-21cm-Intensity-Mapping/blob/a053e4ee5e1aa623e0695f46445bc9334bab2245/9.16/outputs_916/fisher_denominator_comparison.csv)、
  同名 joblib 与 [run_config.json](https://github.com/lhkpai/Probing-Primordial-non-Gaussianity-from-Reconstructing-21cm-Intensity-Mapping/blob/a053e4ee5e1aa623e0695f46445bc9334bab2245/9.16/outputs_916/run_config.json)；对应 GitHub 提交
  [`a053e4e`](https://github.com/lhkpai/Probing-Primordial-non-Gaussianity-from-Reconstructing-21cm-Intensity-Mapping/commit/a053e4ee5e1aa623e0695f46445bc9334bab2245)。

核对了三组共 **30 组 f_NL 结果**，并从已有矩阵复算了
**240 组七参数误差向量**（四类矩阵、含／不含 CMB），与保存的约束表一致。
当前 CSV 与 joblib 一致；除 `kmax`、`kpar_max` 外的配置一致；
五套体素标定、Noise 模块及望远镜配置文件的记录哈希一致，Noise 数据文件的记录路径、大小和修改时间一致。

本文覆盖最近三组使用同一完整 Noise 表的双分母运行。更早总 k 上限 0.535 的阶段伴随 Noise 接入等其他变更，未混入本次截断对照。
本文仅整理已有结果与模式范围，不重新计算 Fisher 或体素方差。
