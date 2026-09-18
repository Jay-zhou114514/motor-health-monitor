# 数据布局（Data Layout）

- 更新：2026-09-18
- 原因：C 盘容量有限（Paderborn 解压后约 5 GB，XJTU-SY 预计 10 GB 量级），
  大数据集统一放在 **E 盘**，仓库内只保留小数据集与代码。

## 1. 位置

| 数据集 | 路径 | 体积（近似） | 说明 |
| --- | --- | --- | --- |
| **外部数据根** | `E:\MotorHealthMonitorData` | — | 由 `src/config.py` 的 `EXTERNAL_DATA_ROOT` 指定 |
| Paderborn | `E:\MotorHealthMonitorData\paderborn\` | 5.0 GB | `K001–K006.rar` + `extracted/`（480 条记录） |
| PRONOSTIA / FEMTO-ST | `E:\MotorHealthMonitorData\pronostia\` | 0.7 GB（zip）+ 解压后更多 | `phm2012.zip` + `extracted/`（17 颗轴承） |
| XJTU-SY | `E:\MotorHealthMonitorData\xjtu_sy\XJTU-SY_Bearing_Datasets\` | **11.38 GB**（已解压） | 15 颗轴承 × 3 工况；CSV 带表头 |
| IMS | 仓库内 `data/raw/ims/` | 22 MB | V1 全部实验用 |
| MFPT（MathWorks） | 仓库内 `data/raw/{train,test}_data/` | 21 MB | — |

## 2. 代码如何找到数据

`src/config.py`：

```python
EXTERNAL_DATA_ROOT = Path(r"E:\MotorHealthMonitorData")
DATASET_ROOT = EXTERNAL_DATA_ROOT if EXTERNAL_DATA_ROOT.exists() else RAW_DATA_DIR
```

- 外部目录存在 → 用 E 盘；
- 不存在（例如换机器/换盘符）→ **自动回退**到仓库内 `data/raw`，代码仍可运行；
- 因此**仓库本身不依赖 E 盘**，但复现大数据实验时需要把数据放到上述位置。

## 3. 为什么数据不入库

`.gitignore` 已排除 `data/`：体积过大，且多数数据集有独立的许可与引用要求
（Paderborn 需引用原始数据集论文；XJTU-SY 要求引用 Wang et al., IEEE Trans. Reliability, 2020）。
**论文与文档必须记录数据来源与引用，而不是把数据本身提交到仓库。**

## 4. 迁移记录

| 日期 | 动作 |
| --- | --- |
| 2026-09-18 | Paderborn（4.97 GB）与 PRONOSTIA zip 从仓库 `data/raw/` 迁移至 `E:\MotorHealthMonitorData\`；`config.py` 与 `paderborn_data.py` 同步更新 |