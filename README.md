## 用途

本项目是「激光雷达点云测量建模平台」的代码仓库，用于逐步实现该方向的建模与数据处理能力。

本项目用于逐步实现激光雷达点云建模与数据处理能力。

目前已实现体素级融合算法 `lidar_scan.voxel.fuse_voxels`。

## 环境与安装

- Python 3.11 及以上

```bash
python -m pip install -e .
```

## 测试

```bash
python -m pytest
```

基线尚无测试用例，收集到 0 个用例属预期结果。

## 命令行入口

安装后提供 `lidar-scan-modeling` 命令：

```bash
lidar-scan-modeling version    # 打印版本号
lidar-scan-modeling --help     # 打印用法
```

## 现有公开接口

- 命令行程序 `lidar-scan-modeling`
- Python 包 `lidar_scan`，其 `__version__` 为当前版本号
- `lidar_scan.voxel.fuse_voxels(points, voxel_size=1.0)`：体素级点融合

### `fuse_voxels`

- 入参 `points` 为可迭代对象，每个点必须是五项 tuple/list：`(x, y, z, intensity, sigma)`；五项及 `voxel_size` 均须为非 bool 的 int/float。
- 各坐标除以 `voxel_size` 后向负无穷取整得到体素索引。
- 同一体素内以 `w = 1 / sigma²` 为权求 x、y、z、intensity 的加权均值，融合后的 sigma 为 `sqrt(1 / sum(w))`。
- 返回 tuple，内层 tuple 九项依次为 `ix, iy, iz, x, y, z, intensity, sigma, count`，按前三项字典序排列；空输入返回 `()`。
- 全部数值运算使用 Decimal（精度 50，ROUND_HALF_EVEN），结果与输入顺序无关；五个输出小数量化至六位后转 float，负零归一为正零。
- 非法输入抛 `TypeError`（不可迭代、点不是 tuple/list、长度不符、数值类型非法）；`voxel_size <= 0`、`sigma <= 0` 或数值非有限抛 `ValueError`。

```python
from lidar_scan.voxel import fuse_voxels

fuse_voxels([(0.0, 0.0, 0.0, 10.0, 1.0), (0.5, 0.5, 0.5, 20.0, 1.0)])
# ((0, 0, 0, 0.25, 0.25, 0.25, 15.0, 0.707107, 2),)
```

## 限制

- 除版本查询与体素融合外没有其他功能。
- 其他输入输出格式、数据来源与算法均尚未定义。
