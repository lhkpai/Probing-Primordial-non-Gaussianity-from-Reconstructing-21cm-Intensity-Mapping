# 9.16 修改与物理核对（2026-09-20 更新）

已在干净 Python 内核中运行完整 notebook。体素方差上限现改为0.363 Mpc⁻¹，
Fisher上限仍为0.535 Mpc⁻¹；体素上限修改记录见 `voxel_kmax_update.json`；最新 Δz=0.01 运行检查见 `delta_z_update.json`。数值结果保存在本目录；
旧 notebook 和原版 baofisher_k 已备份在 `backups/`。

## 实现对应关系

| 用户需求 | 实现及结果 |
|---|---|
| 五个指定干涉配置 | SKA-MID、CHIME、HIRAX、Tianlai、MeerKAT，均已运行 |
| 红移箱宽 0.01 | 固定步长，保留指定输出边界及原有末箱处理 |
| 原版横向、径向下限 | 保留基线文件/均匀基线、适用的 FoV 及前景截止 |
| 独立总波数上限 | 体素方差 `voxel_kmax=0.363`；Fisher `kmax=0.535`，不联动关闭其他限制 |
| 热噪声体素方差 | 使用 baofisher 的热噪声谱，不加入 Cfg，不再调用单碟辅助方差 |
| 3000 Mpc/h、512 网格、sinc 窗口 | 盒长按 h_mock=0.6766 换算，体素边长 8.660028 Mpc |
| 柱坐标 Fisher | 使用 k_perp、正 k_parallel 与正确积分测度；正负模式权重明确 |
| mock 原二维坐标、单位一致 | 波数由明确配置的 h/Mpc 转为 Mpc⁻¹，三维线性插值(z,p,ell) |
| 暂不修正模式数归一化 | 保留额外分母 fisher_arr × P_HI²，并在元数据注明 |
| 保存方差文件 | 每配置 CSV/joblib，加全部配置 CSV 合表 |

主 notebook 负责配置、运行和展示；`../forecast_916.py` 放置可测试的预测流程，
`../voxel_modes_916.py` 放置有限盒子模式求和。原版 `baofisher.py` 未改动。
`baofisher_k.py` 增加 `noise_k_limits` 和 `interferometer_mode_mask`，
使物理覆盖可以在积分前明确检查，而不靠极大噪声数值判断。

## 积分和方差定义

热噪声谱采用原有转换：

$$P_N(p,\ell,z)=r^2r_\nu C_N(rp,r_\nu\ell,z),\qquad
r_\nu=c(1+z)^2/H(z).$$

原版前景截断仍是模式选择，输出不包含前景残差协方差。Tsky 保留在系统温度中。

### 为什么采用有限盒子的模式和

CHIME/Tianlai 的部分相邻基线采样点从零密度变为正密度。原版线性插值使
边缘附近 n(p) 正比于 p-p0，因此连续体素积分出现 ∫p dp/(p-p0) 的对数发散。
单个正密度点的噪声仍可有限，原版 Fisher 也不会因此发散，因为其权重随 n² 趋零。

本实现不改变这些基线边界，不设置人为密度下限。采用用户指定周期盒子的有限模式：

$$\sigma_N^2=L_{\rm Mpc}^{-3}\sum_{\mathbf k\in\mathcal D}
P_N(\mathbf k)|W_{\rm pix}(\mathbf k)|^2,\qquad
\mathbf k=2\pi\mathbf n/L_{\rm Mpc}.$$

$$|W_{\rm pix}|^2=\prod_i\operatorname{sinc}^2(k_i\Delta/2),\qquad
\Delta=L_{\rm Mpc}/512.$$

这是一项明确的有限盒子定义，不是发散连续积分的收敛值。横向整数半径相同的模式
按精确简并数和窗口权重合并；径向利用原版频道响应的可分离性作累计和。
它与直接三维求和等价。盒长决定模式间隔，改变盒长可能改变基线边缘贡献。

N=512 定义体素平均宽度。本次仅将方差的总波数上限改为0.363，
候选模式随后应用原版仪器与前景掩码。没有改成512³ FFT数组枚举：
当前球内最大轴向索引为256，正负Nyquist边界均保留；
同时球外的FFT对角模式被排除。该球形截断不能等同于完整FFT立方体。

每个红移箱的输出是配置的有限红移节点上局部方差的体积加权平均；
不是把整箱温度平均成一个体素后的方差。观测时间不按红移箱数拆分。

### Fisher

正径向半轴的归一化为：

$$F_{ij}=\int dz\,\frac{dV/dz}{4\pi^2}
\int p\,dp\,d\ell\,
\frac{P_{,i}P_{,j}}{(P_{\rm HI}+P_N)^2+D_{\rm mock}}.$$

解析取 D_mock=0；叠加 mock 时暂用 D_mock=f_mock P_HI²。
体素 sinc 窗口不单独乘进 Fisher 分母；体素方差也不替代逐模式噪声谱。

横向积分在基线文件的插值节点处分段，避免全局网格漏掉圆柱阵列的密度变化。
CLASS 宇宙学在所有红移间复用；原有信号模型、参数步长和 CMB 先验保留。
Fisher 采用尺度归一化后的正定性检查和求逆，不再用 abs(diag(cov)) 隐藏异常。

## 红移、mock 和剩余物理假设

- 输出仍覆盖旧红移边界，但真实频段外不能外推仪器观测。最新分箱数量见
  `execution_summary.json`；频段外保存 NaN。Fisher只使用实际覆盖体积。
- mock 的六层红移采用原始产品路径标签：0.511、0.989、1.501、2.005、2.501、3.017。
  来源为 `../variance_audit_20260915/mock_paths.json`，不再用名义 linspace(0.5,3,6)。
- 旧 mock 导出文件缺乏单位元数据。本次沿用已讨论的 h/Mpc 与 h_mock=0.6766，
  将该假设作为显式配置保存，而不换用 Fisher 宇宙学 h≈0.67197。
- mock 域外不填零、不夹取红移端点。`sigma_ana_common` 和 `sigma_ana_mock`
  使用相同的模式与红移域；`sigma_ana_full` 是完整仪器分析域的独立参考。
- HIRAX基线文件仍缺失，使用6–200 m均匀基线模型。Tianlai继续使用原notebook的圆柱几何补充参数。
- 模式数归一化、原mock噪声档位平均、不同仪器的恢复误差标定未在此次修改。
  因此解析加mock是当前指定方差模型的结果，不应称为已经完整校准的实际恢复约束。

## 验证结果

`verify_916.py` 与 `validation.json` 保存原始数值算法验证；以下精度数字来自该历史验证。
当前分箱更新检查见 `check_dz001.py`、`delta_z_update.json`。

- 未指定新上限时，新旧 Cnoise 在五个配置的验证网格上逐元素一致。
- 有限盒子分组求和与独立三维暴力求和相对误差约1.1×10⁻¹⁶；模式数完全一致。
- 常量谱的柱坐标与球坐标 Fisher 归一化相对误差约3.2×10⁻⁸。
- 五个配置观测时间增加十倍，体素方差均降为十分之一。
- 在 z=1 将 Fisher 积分精度加倍，所有参数误差最大变化小于0.07%，
  f_NL误差最大变化小于0.022%。这是一组代表红移的检查，不是所有模型系统误差的界限。
- 信号谱插值对直接 CLASS 结果的检查误差小于1.3×10⁻⁸。
- mock插值节点值一致，域外查询被拒绝；加入非负mock项不增加Fisher信息。
- 整本notebook在干净内核完成执行，零异常输出；正式约束表全部为有限值。

## 主要输出

- `voxel_noise_all_surveys.csv`：全部方差结果。
- `voxel_noise_<survey>.csv/.joblib`：每配置方差及元数据。
- `constraints_<survey>.csv`：三种域/方差设置及各自联合CMB的参数误差。
- `fisher_<survey>.joblib`：Fisher矩阵和参数顺序。
- `voxel_noise_all_surveys.png`：噪声方差和有效基线覆盖比例。
- `execution_summary.json`：整本运行和输出检查。
