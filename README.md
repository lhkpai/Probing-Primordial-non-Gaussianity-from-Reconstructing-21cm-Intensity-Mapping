# Probing-Primordial-non-Gaussianity-from-Reconstructing-21cm-Intensity-Mapping

bao21cm_master + 6.7-interferometer-telescopes-2 + fisher_1.3.2.1_DirctTF_GenNoise_AssisHKL_Tutorial

## 仓库结构
- 项目代码：`6.7-interferometer-telescopes-2.ipynb` — 多望远镜 21 cm IM+CM 联合约束 f_NL Fisher（解析 + mock）
  - 噪声功率谱等函数调用了`baofisher_k.py`
  - 望远镜配置来自`experiments.py`
  - 栾天成模拟的mock方差数据来自`fisher_1.3.2.1_DirctTF_GenNoise_AssisHKL_Tutorial.ipynb`
- `bao21cm_master/` — baofisher 21 cm intensity-mapping Fisher forecast 代码库
  - `radiofisher/` — 核心库（`baofisher.py`/`baofisher_k.py`/`experiments.py`/`camb_wrapper.py` 等）
  - `array_config/` — 各干涉仪基线密度 n(u) 数据文件（运行必需）
  - `plotting/`、`output/` — 绘图脚本与示例结果
  - 顶层 `*.py` — 各类巡天灵敏度/Fisher 计算与测试脚本
- `fisher_1.3.2.1_DirctTF_GenNoise_AssisHKL_Tutorial.ipynb` — 栾天成模拟的mock方差数据
- `MEMO_6.7_interferometer_fNL_forecast.md` - `6.7-interferometer-telescopes-2.ipynb`代码逻辑说明，与AI对话总结

## 使用
详见 `MEMO_6.7_interferometer_fNL_forecast.md`、`bao21cm_master/README`、`bao21cm_master/REPRODUCE_PAPER.sh`；notebook 内注释含关键物理参数与推导。
