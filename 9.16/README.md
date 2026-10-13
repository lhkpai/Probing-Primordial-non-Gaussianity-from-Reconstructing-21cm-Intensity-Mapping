# 9.16：体素热噪声方差与 Fisher 预测

日常入口是 [9.16-interferometer-telescopes.ipynb](9.16-interferometer-telescopes.ipynb)。
Notebook 调用 `run_fisher_only(cfg, SURVEYS)`，复用现有体素方差和 Noise 标定表。
它按当前分母更新 Fisher 基准，并同时计算线性与 spline；匹配结果存在时直接读取。
随后调用 `compare_fisher_denominators(cfg, SURVEYS)`，并列展示解析项含/不含 P_N 的两种模型。
该入口复用已有不含 P_N 的结果，只补齐缺失的含 P_N 模型，之后读取同一比较缓存。
当前配置为 `delta_z=0.01`、`kmax=voxel_kmax=0.363 Mpc^-1`。

## 文件作用

| 位置 | 用途 |
| --- | --- |
| `forecast_916.py` | 组织信号、仪器噪声、新 Noise 接入、Fisher 和结果保存 |
| `voxel_modes_916.py` | 有限盒子模式求和与三维 sinc 体素窗口 |
| `diagnose_noise_916.py` | 红移平滑性和 k 下限增量诊断 |
| `tests/verify_916.py` | 模式求和、坐标测度和预测流程的数值验证 |
| `tests/check_common_kmax.py` | 核对保存结果、共同域、分母模型和体素方差一致性 |
| `tools/execute_notebook.py` | 在干净内核中执行 notebook 并保存输出 |
| `outputs_916/` | 当前 notebook 的唯一实际结果目录 |

## 输入、结果与共享依赖

- 新 Noise 数据为
  `/mnt/xiaodong/luantch/LSS21cm_freeloadXD/PosiRestore/FisherMess/noise_hkl/data/noise_hkl_N200.h5`。
  程序复用同一标定目录内的 `noise_hkl.py`，旧 mock 不再参与计算。
- 传递函数缓存仍为 `/home/liuhk/research/transfer_bf_cache.dat`。
- 当前结果只保存在 `/home/liuhk/research/9.16/outputs_916/`；该目录是实际目录。
- `/home/liuhk/research/outputs_6.7/` 保留旧 6.7 结果，与当前 9.16 结果分开。
- `run_config.json` 是保存的运行参数记录；参数入口在 notebook。
- `baofisher_k.py`、`experiments.py`、其他 RadioFisher 模块和基线数据均保留在
  `/home/liuhk/bao21cm-master/bao21cm-master/` 下的共享版本。
  程序通过原 `radiofisher` 包导入这些模块。

## Noise 与 Fisher

先在标定分箱上形成 `A = N_cal * tf_var / (2 * tf_mean**2)`，再插值。
读取器已完成该转换；Fisher 直接使用返回的 `A_noise_conditional`：

```python
denom_ana = p_hi**2
denom_noise = p_hi**2 * a_noise
denom_total = denom_ana + denom_noise
```

每个望远镜读取已有体素方差，沿 `z_mid` 插值方差并取平方根，取得查询 RMS。
插值在 `(z, log sigma)` 和 `(k_perp, k_parallel)` 上完成，波数单位显式转换。
两种对照使用同一有效域；`sigma_ana_full` 单独保留完整仪器域解析约束。

默认 `noise_fisher_only=False`，使用完整200×200 Noise表的数值有效部分。
实际积分域仍与仪器覆盖及 `k<=0.363 Mpc^-1` 取交集，不作表外外推。
原39个低k箱来自 `k_eff=0.025 h/Mpc` 的筛选，不是完整表的有效性边界。
本次 spline 对比需要完整表。完整箱间协方差目前仅覆盖原39箱；
当前两种设置均只使用逐箱方差，因此完整表预测仍采用对角协方差近似。
无支持点排除并记录；空交集的对照约束为 NaN，CMB先验不会掩盖该状态。

默认模型的解析项只保留宇宙方差，额外恢复散布由条件 Noise 项提供。
白噪声 RMS 匹配不等于完整仪器噪声谱匹配；`1/N` 迁移、箱间相关和恢复后的参数响应需要独立验证。

新增对照将解析分母改为 `(p_hi + p_noise)**2`，额外 Noise 项仍为 `p_hi**2 * a_noise`。
两种解析分母使用相同的体素 RMS、标定表、波数/红移域、参数导数和 CMB 先验。
“解析项不含 P_N”不代表整套恢复模型没有噪声，条件 Noise 项仍然保留。

五个配置使用完整表，均有有效共同支持。当前 notebook 配置、表格和序列化结果
由 `tests/check_common_kmax.py` 核对，该检查不依赖历史备份。
整本运行和检查结果见 `outputs_916/execution_summary.json`；物理定义与数值结果
见 `outputs_916/implementation_notes.md`。
结果覆盖信息见 `noise_coverage_all_surveys.csv`、`noise_calibration_summary.json`。
`noise_diagnostics/` 保留当前热噪声模型的红移与 k 下限诊断。
只保留当前代码和当前结果；重复源码、历史生成工具及旧 9.16 备份已清理。

## 线性与三次样条对比

`run_fisher_only` 读取现有体素表，更新完整仪器域解析基准，
共同域只计算一遍，共用信号导数和积分节点，同时得到解析、双线性与三次样条预测。
Notebook 从 `compare_fisher_denominators` 的统一结果展示两种分母各自的插值对照。
`compare_noise_interpolations` 仍可单独读取默认模型的插值结果。

- 两者均先计算原始分箱的 `A=N_cal*tf_var/(2*tf_mean**2)`。
- `(z, log sigma)` 仍使用相同的 Delaunay 线性权重。
- 三次样条为 `RectBivariateSpline(kx=3, ky=3, s=0)`，在完整有效的二维 A 面上构造。
- 表外不外推；负值或非有限样条值明确回退到原线性值，保留相同模式域。
- 在原结果目录更新 `constraints_<survey>.csv` 和基准 Fisher；体素 CSV/joblib 保持不变。

这里插值的 A 是按模式数归一化的相对方差，无量纲；它描述额外恢复散布随尺度的变化。
双线性插值取相邻四点的非负权重平均，假设箱内近似线性变化。
三次样条用光滑曲面连接原始点，允许弯曲，也可能越过相邻数据的取值范围。
`s=0` 表示穿过原始点，并不消除标定值本身的统计散布。
两者的差异用于检查预测对插值选择的敏感性，不能据光滑程度判断物理准确性。

插值及分母对照保存在以下固定结果文件中：

| 文件 | 内容 |
| --- | --- |
| `noise_interpolation_comparison.csv` | 五个望远镜的 f_NL 误差、相对变化、CMB 组合和回退权重占比 |
| `noise_interpolation_comparison.joblib` | 三组共同域矩阵、全部参数误差、检查信息和输入指纹 |
| `fisher_denominator_comparison.csv` | 两种解析分母、五个配置、七个参数，共70行的全部约束 |
| `fisher_denominator_comparison.joblib` | 两种模型的完整/共同域矩阵、线性/spline与CMB组合及输入指纹 |

输入及相关计算代码未变时直接读取对应缓存，不重复计算。
需要明确强制重算对比时可使用 `reuse_saved=False`。

## 验证与重新运行

```bash
/software/anaconda3/bin/python3 tools/execute_notebook.py > outputs_916/execution_notebook.log 2>&1
/software/anaconda3/bin/python3 tests/verify_916.py --fisher-only
/software/anaconda3/bin/python3 tests/check_common_kmax.py
```

以上命令在本目录运行。日常使用仍只需运行 notebook。
