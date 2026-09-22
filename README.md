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
- `lidar_scan.chm.build_chm(points, ground, cell_size=1.0)`：基于地面格网构建冠层高度模型
- `lidar_scan.las.iter_las(path)`：分块流式读取 LAS/LAZ 激光雷达文件
- `lidar_scan.ply.iter_ply(path, chunk_size=65536)`：分块流式读取 PLY 激光雷达文件
- `lidar_scan.e57.iter_e57(path, chunk_size=65536)`：分块流式读取 E57 激光雷达文件

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

### `build_chm`

- `points`、`ground` 均为可迭代对象且只迭代一次（可为不支持 `len` 的迭代器）；每个点为五项 tuple/list `(x, y, z, intensity, sigma)`，每个地面格为五项 tuple/list `(ix, iy, z, sigma, count)`
- 点的坐标/强度/z/sigma 及 `cell_size` 须为非 bool 的 int/float 且有限；地面的 `ix`、`iy`、`count` 须为非 bool int；点坐标除以 `cell_size` 后向负无穷取整得到格索引
- 点按 `(floor(x/cell_size), floor(y/cell_size))` 匹配地面格，点格缺少对应地面格抛 `ValueError`；地面格索引重复抛 `ValueError`
- 每格取 z 最高的点，z 并列时依次按 sigma、intensity、x、y 升序择首；输出 `count` 为该格全部点数
- 输出 tuple，每项为五元组 `(ix, iy, height, sigma, count)`，按 `(ix, iy)` 字典序排列；`points` 为空时返回 `()`
- `height = max(0, point_z - ground_z)`，`sigma = sqrt(point_sigma² + ground_sigma²)`；计算使用 `Decimal(str(v))`（精度 50、ROUND_HALF_EVEN），height/sigma 量化到六位小数后转 float，负零归一为正零
- 容器不可迭代、点/地面格容器或长度不符、标量类型（含 `cell_size`）非法抛 `TypeError`；`cell_size` 或任一 sigma 非有限/≤0、地面 `count ≤ 0`、地面索引重复、点格缺地面抛 `ValueError`

```python
from lidar_scan import build_chm

build_chm([(0.2, 0.2, 3.0, 10.0, 0.5)], ground=[(0, 0, 1.0, 0.5, 4)])
# ((0, 0, 2.0, 0.707107, 1),)
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

### `iter_ply`

- `path` 只接受 `str`，且后缀大小写不敏感地必须是 `.ply`
- `chunk_size` 只接受非 bool 的 `int` 且必须为正，默认 65536；`path` 类型错误或 `chunk_size` 类型/取值错误在调用时抛出
- 返回一次性迭代器，按文件记录顺序产出 tuple 块；除末块外每块恰含 `chunk_size` 点，末块为剩余点；空文件不产出任何块
- 分块增量读取，不会整文件载入；正常耗尽、异常或迭代器被回收都会关闭文件句柄，再次迭代不会重开文件
- 仅支持 `format ascii 1.0` 与 `format binary_little_endian 1.0`（不支持 big endian）
- 头部必须恰好有一个非负的 `element vertex N`，不能有任何其他 element；属性仅允许标量 `x/y/z` 及可选 `intensity/sigma`
- 属性类型限 `char/uchar/short/ushort/int/uint/float/double`；list 属性、未知或重复属性、缺少 `x/y/z`、声明数量与记录数不符均抛 `ValueError`
- 按属性声明顺序解析，但每点严格输出五项 tuple `(x, y, z, intensity, sigma)`：
  - `x/y/z` 为数值；`intensity` 必须为整数（浮点属性中也必须是整数值），缺省为 `0`，转 int
  - `sigma` 为数值，必须有限且为正，缺省为 `1.0`
- 坐标与 sigma 经 `Decimal(str(v))`（精度 50、ROUND_HALF_EVEN）量化到六位小数后转 float，负零归一为正零
- 文件无法打开统一抛 `OSError`；头部或点记录非法/截断、坐标非有限、intensity 非整数、sigma 非有限或非正均抛 `ValueError`

```python
from lidar_scan import iter_ply

for block in iter_ply("scan.ply", chunk_size=65536):
    # 每个 block 是 chunk_size 个 (x, y, z, intensity, sigma) tuple（末块除外）
    for x, y, z, intensity, sigma in block:
        ...
```

### `iter_e57`

- `path` 只接受 `str`，且后缀大小写不敏感地必须是 `.e57`
- `chunk_size` 只接受非 bool 的 `int` 且必须为正，默认 65536；`path` 类型错误或 `chunk_size` 类型/取值错误在调用时抛出
- 返回惰性一次性迭代器，按 E57 扫描索引顺序跨扫描拼接、各扫描内按点记录顺序产出 tuple 块；除末块外每块恰含 `chunk_size` 点，末块为剩余点；空文件或全部为零点数扫描不产出任何块
- 分块增量解码，不会整文件载入；正常耗尽、异常或迭代器被回收都会关闭文件句柄，再次迭代不会重开文件
- 每个扫描必须提供 `cartesianX`、`cartesianY`、`cartesianZ` 点字段；缺失笛卡尔坐标（包括仅有球坐标的扫描）抛 `ValueError`；不做位姿变换，直接输出笛卡尔字段值
- 每点统一输出五项 tuple `(x, y, z, intensity, sigma)`：可选 `intensity` 字段缺省为 `0`，`sigma` 固定为 `1.0`
- 坐标经 `Decimal(str(v))`（精度 50、ROUND_HALF_EVEN）量化到六位小数后转 float，负零归一为正零；intensity 必须为有限整数后转 int，坐标必须有限
- 文件无法打开统一抛 `OSError`；未安装 `pye57` 抛 `RuntimeError`；容器、扫描、字段、记录截断或解码失败、非有限值均抛 `ValueError`

```python
from lidar_scan import iter_e57

for block in iter_e57("scan.e57", chunk_size=65536):
    # 每个 block 是 chunk_size 个 (x, y, z, intensity, sigma) tuple（末块除外）
    for x, y, z, intensity, sigma in block:
        ...
```

## 限制

- 除版本查询、LAS/LAZ 分块读取、PLY 分块读取、E57 分块读取、体素融合与冠层高度模型外没有其他功能。
- 其他输入输出格式、数据来源与算法均尚未定义。
