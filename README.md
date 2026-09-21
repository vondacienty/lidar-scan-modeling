## 用途

本项目是「激光雷达点云测量建模平台」的代码仓库，用于逐步实现该方向的建模与数据处理能力。

当前处于基线状态：只有项目骨架，尚未实现任何业务算法。

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
- `lidar_scan.voxel.fuse_voxels(points, voxel_size=1.0)`：按体素融合激光雷达点

### `fuse_voxels`

- `points` 为可迭代对象，每个点必须是五项 tuple/list：`(x, y, z, intensity, sigma)`
- 五项及 `voxel_size` 均须为非 bool 的 int/float；坐标除以 `voxel_size` 后向负无穷取整得到体素索引
- 同一体素内以 `w = 1 / sigma²` 为权重对 x、y、z、intensity 求加权均值，融合 `sigma = sqrt(1 / sum(w))`
- 返回 tuple，每项为九元组 `(ix, iy, iz, x, y, z, intensity, sigma, count)`，按 `(ix, iy, iz)` 字典序排列；空输入返回 `()`
- 计算使用 `Decimal`（精度 50、ROUND_HALF_EVEN），结果与输入次序无关；五个输出小数保留六位并转为 float，负零归一为正零
- `points` 不可迭代、点容器或长度不符、参数类型非法抛 `TypeError`；`voxel_size <= 0`、`sigma <= 0` 或数值非有限抛 `ValueError`

```python
from lidar_scan import fuse_voxels

fuse_voxels([(0.0, 0.0, 0.0, 0.0, 1.0),
             (0.5, 0.5, 0.5, 10.0, 0.5)])
# ((0, 0, 0, 0.4, 0.4, 0.4, 8.0, 0.447214, 2),)
```

## 限制

- 除版本查询与体素融合外没有其他功能。
- 其他输入输出格式、数据来源与算法均尚未定义。
