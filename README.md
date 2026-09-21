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
- `lidar_scan.las.iter_las(path)`：分块流式读取 LAS/LAZ 激光雷达文件

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

### `iter_las`

- `path` 只接受 `str`，且后缀大小写不敏感地必须是 `.las` 或 `.laz`
- 返回一次性迭代器，按文件记录顺序产出 tuple 块；除末块外每块恰含 65536 点，末块为剩余点；空文件不产出任何块
- 分块增量读取，不会整文件载入；正常耗尽、异常或迭代器被回收都会关闭文件句柄，再次迭代不会重开文件
- 每点为五项 tuple `(x, y, z, intensity, sigma)`：
  - `x/y/z = 原始整数 X/Y/Z × 各轴 scale + offset`
  - `intensity` 取标准 LAS 字段并转 int
  - 存在名为 `sigma` 的标量数值额外维度时取其（含额外字节 scale/offset 的）解码值，否则固定为 `1.0`
- 坐标计算使用 `Decimal(str(v))`、精度 50、ROUND_HALF_EVEN；`x/y/z/sigma` 量化到六位小数后转 float，负零归一为正零
- 参数错误在调用时抛出：`path` 非 `str` 抛 `TypeError`，后缀不支持抛 `ValueError`；文件无法打开统一抛 `OSError`，LAZ 解码器（lazrs）不可用抛 `RuntimeError`
- 文件头或点记录不可解析、必需标准字段缺失、scale/offset 非有限、sigma 非有限或非正均抛 `ValueError`

```python
from lidar_scan import iter_las

for block in iter_las("scan.laz"):
    # 每个 block 是 65536 个 (x, y, z, intensity, sigma) tuple（末块除外）
    for x, y, z, intensity, sigma in block:
        ...
```

## 限制

- 除版本查询、LAS/LAZ 分块读取与体素融合外没有其他功能。
- 其他输入输出格式、数据来源与算法均尚未定义。
