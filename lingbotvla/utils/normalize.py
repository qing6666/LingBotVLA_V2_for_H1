# -*- coding: utf-8 -*-
"""
normalize.py —— 归一化统计量的"算法内核"
====================================================================
本文件提供三类东西:
  1) NormStats        :最终统计量容器(mean/std/q01/q99/min/max ...),会被序列化进 JSON。
  2) RunningStats     :流式(在线)统计器。数据集太大无法全载入内存,所以逐 batch
                       在线累计 mean / 方差 / 极值 / 直方图(用于算分位数)。
                       被 scripts/compute_norm_stats.py 调用。
  3) save / load 等   :统计量的 JSON 序列化与反序列化。
  4) RunningStatsState:RunningStats 内部状态的序列化模型,用于"跨卡合并"时传递。
====================================================================
"""

import json
import pathlib

import numpy as np
import numpydantic
import pydantic


@pydantic.dataclasses.dataclass
class NormStats:
    """最终归一化统计量(每个特征一份)。训练时 Normalizer 根据这些值做归一化。"""
    mean: numpydantic.NDArray
    std: numpydantic.NDArray
    q01: numpydantic.NDArray | None = None  # 1st quantile  1% 分位数(bounds_99 归一化的下界)
    q99: numpydantic.NDArray | None = None  # 99th quantile 99% 分位数(bounds_99 归一化的上界)
    q02: numpydantic.NDArray | None = None  # 2nd quantile  2% 分位数(bounds_98 归一化的下界)
    q98: numpydantic.NDArray | None = None  # 98th quantile 98% 分位数(bounds_98 归一化的上界)
    min: numpydantic.NDArray | None = None  # 最小值(原英文注释误写为 1st quantile,实际是 min;用于 minmax)
    max: numpydantic.NDArray | None = None  # 最大值(原英文注释误写为 99th quantile,实际是 max;用于 minmax)


class RunningStatsState(pydantic.BaseModel):
    """Model for persisting the internal state of RunningStats"""
    """RunningStats 内部状态的序列化模型(用于跨卡 all_gather 时传递、或存盘续算)。"""
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    count: int
    mean: numpydantic.NDArray
    mean_of_squares: numpydantic.NDArray
    min_val: numpydantic.NDArray
    max_val: numpydantic.NDArray
    histograms: numpydantic.NDArray  # Shape: (vector_length, num_bins)   每个特征维一条 5000 格直方图
    bin_edges: numpydantic.NDArray   # Shape: (vector_length, num_bins + 1) 每个特征维的分箱边界
    num_quantile_bins: int
class RunningStats:
    """Compute running statistics of a batch of vectors."""
    """流式统计器:逐 batch 在线累计一个向量序列的统计量,无需保存全部原始数据。"""

    def __init__(self):
        self._count = 0              # 已累计的样本数
        self._mean = None            # 在线均值 E[x]
        self._mean_of_squares = None # 在线 E[x²](用来算方差:Var = E[x²] - E[x]²)
        self._min = None             # 逐维最小值
        self._max = None             # 逐维最大值
        self._histograms = None      # 每维一条直方图(算分位数用)
        self._bin_edges = None       # 每维的分箱边界
        self._num_quantile_bins = 5000  # for computing quantiles on the fly  分位数直方图的格数

    def update(self, batch: np.ndarray) -> None:
        """
        Update the running statistics with a batch of vectors.
        用一个 batch 的向量更新在线统计量(核心算法)。

        Args:
            vectors (np.ndarray): A 2D array where each row is a new vector.
                                  (N, D) 的数组,每行一个 D 维样本。
        """
        if batch.ndim == 1:
            batch = batch.reshape(-1, 1)   # 1D 自动转成 (N,1)

        num_elements, vector_length = batch.shape   # N 个样本,每个 D 维

        if self._count == 0:
            # ---- 第一个 batch:初始化所有统计量 ----
            self._mean = np.mean(batch, axis=0)            # 逐维均值
            self._mean_of_squares = np.mean(batch**2, axis=0)  # 逐维 E[x²]
            self._min = np.min(batch, axis=0)
            self._max = np.max(batch, axis=0)
            # 每维建一条全零直方图 + 对应分箱边界(略加 1e-10 padding 防 0 宽区间)
            self._histograms = [np.zeros(self._num_quantile_bins) for _ in range(vector_length)]
            self._bin_edges = [
                np.linspace(self._min[i] - 1e-10, self._max[i] + 1e-10, self._num_quantile_bins + 1)
                for i in range(vector_length)
            ]
        else:
            # ---- 后续 batch:校验维度一致 ----
            if vector_length != self._mean.size:
                raise ValueError("The length of new vectors does not match the initialized vector length.")
            new_max = np.max(batch, axis=0)
            new_min = np.min(batch, axis=0)
            max_changed = np.any(new_max > self._max)
            min_changed = np.any(new_min < self._min)
            self._max = np.maximum(self._max, new_max)   # 更新全局极值
            self._min = np.minimum(self._min, new_min)

            if max_changed or min_changed:
                self._adjust_histograms()   # 极值变化 → 把旧直方图重新分箱到新区间

        self._count += num_elements   # 累计样本数

        batch_mean = np.mean(batch, axis=0)
        batch_mean_of_squares = np.mean(batch**2, axis=0)

        # Update running mean and mean of squares.
        # 在线更新公式(Welford 思想):用本 batch 均值修正累计均值,权重 = n / 总数
        self._mean += (batch_mean - self._mean) * (num_elements / self._count)
        self._mean_of_squares += (batch_mean_of_squares - self._mean_of_squares) * (num_elements / self._count)

        self._update_histograms(batch)   # 把本 batch 投到直方图里

    def get_statistics(self, chunk_size=None) -> NormStats:
        """
        Compute and return the statistics of the vectors processed so far.
        导出最终的统计量。

        Returns:
            dict: A dict containing the computed statistics.
        """
        if self._count < 2:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")

        variance = self._mean_of_squares - self._mean**2      # Var = E[x²] - E[x]²
        stddev = np.sqrt(np.maximum(0, variance))             # std;max(0,·) 防数值误差导致负方差
        q01, q99 = self._compute_quantiles([0.01, 0.99])      # 查直方图得 1%/99% 分位
        q02, q98 = self._compute_quantiles([0.02, 0.98])      # 2%/98% 分位

        if chunk_size is not None:
            # 对"相对动作"特征:把逐维统计量 reshape 成 (chunk, feat/chunk) 以匹配动作序列形状
            assert isinstance(chunk_size, int)
            self._mean = self._mean.reshape(chunk_size, -1)
            self._min = self._min.reshape(chunk_size, -1)
            self._max = self._max.reshape(chunk_size, -1)
            stddev = stddev.reshape(chunk_size, -1)
            q01 = q01.reshape(chunk_size, -1)
            q99 = q99.reshape(chunk_size, -1)
            q02 = q02.reshape(chunk_size, -1)
            q98 = q98.reshape(chunk_size, -1)

        return NormStats(mean=self._mean, std=stddev, q01=q01, q99=q99, q02=q02, q98=q98, min=self._min, max=self._max)

    def _adjust_histograms(self):
        """Adjust histograms when min or max changes."""
        """极值变化时:把旧分箱里的计数重新分配到新区间(保持总量不变)。"""
        for i in range(len(self._histograms)):
            old_edges = self._bin_edges[i]
            new_edges = np.linspace(self._min[i], self._max[i], self._num_quantile_bins + 1)

            # Redistribute the existing histogram counts to the new bins
            # 用旧边界点作为"样本",按权重=旧计数,重新落到新分箱里
            new_hist, _ = np.histogram(old_edges[:-1], bins=new_edges, weights=self._histograms[i])

            self._histograms[i] = new_hist
            self._bin_edges[i] = new_edges

    def _update_histograms(self, batch: np.ndarray) -> None:
        """Update histograms with new vectors."""
        """把本 batch 每一维的值投到对应直方图里(逐维计数)。"""
        for i in range(batch.shape[1]):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist

    def _compute_quantiles(self, quantiles):
        """Compute quantiles based on histograms."""
        """根据直方图的累积分布,反查指定分位点(如 1%/99%)对应的数值。"""
        results = []
        for q in quantiles:
            target_count = q * self._count          # 目标累计计数 = 分位比例 × 总样本数
            q_values = []
            for hist, edges in zip(self._histograms, self._bin_edges, strict=True):
                cumsum = np.cumsum(hist)            # 累积分布
                idx = np.searchsorted(cumsum, target_count)  # 找到累计计数首次 >= target 的格子
                q_values.append(edges[idx])         # 该格子对应的边界值即为分位点
            results.append(np.array(q_values))
        return results

    def get_state(self) -> RunningStatsState:
        """Get all current internal states"""
        """导出内部状态(用于跨卡传输/存盘)。"""
        if self._count == 0:
            raise ValueError("No data processed yet.")
        return RunningStatsState(
            count=self._count,
            mean=self._mean,
            mean_of_squares=self._mean_of_squares,
            min_val=self._min,
            max_val=self._max,
            histograms=np.stack(self._histograms, axis=0),   # list[D] -> (D, num_bins)
            bin_edges=np.stack(self._bin_edges, axis=0),
            num_quantile_bins=self._num_quantile_bins
        )

    @classmethod
    def from_state(cls, state: RunningStatsState):
        """Restore a RunningStats object from its state"""
        """从状态对象恢复一个 RunningStats(get_state 的逆操作)。"""
        instance = cls()
        instance._num_quantile_bins = state.num_quantile_bins
        instance._count = state.count
        instance._mean = np.asarray(state.mean)
        instance._mean_of_squares = np.asarray(state.mean_of_squares)
        instance._min = np.asarray(state.min_val)
        instance._max = np.asarray(state.max_val)
        # After numpydantic serialization, histograms/bin_edges become a single 2D array.
        # Internally we split it back into a list[1D-array] per dim, so that
        # _update_histograms / _adjust_histograms can be reused.
        # 序列化后 histograms/bin_edges 变成单个 2D 数组,这里拆回 list[1D],复用上面的逐维方法。
        hist = np.asarray(state.histograms)
        edges = np.asarray(state.bin_edges)
        instance._histograms = [hist[i] for i in range(hist.shape[0])]
        instance._bin_edges = [edges[i] for i in range(edges.shape[0])]
        return instance

    @classmethod
    def merge(cls, others: list["RunningStats"]) -> "RunningStats":
        """Merge multiple RunningStats (typical use: aggregating across ranks).

        Merge formula (per-dim):
            count = Σ cᵢ
            mean = Σ cᵢ·meanᵢ / count
            mean_of_squares = Σ cᵢ·msᵢ / count
            min/max = elementwise min/max
            histograms = rebin each shard's histogram onto unified new_edges, then sum
        """
        """合并多个 RunningStats(典型用途:多卡各算一份后聚合)。
        公式:count 求和;mean / mean_of_squares 按 count 加权平均;min/max 逐元素取极值;
              直方图各自 rebin 到统一的新分箱后相加。"""
        valid = [o for o in others if o is not None and o._count > 0]
        if not valid:
            raise ValueError("merge() requires at least one non-empty RunningStats.")
        if len(valid) == 1:
            return valid[0]

        num_bins = valid[0]._num_quantile_bins
        assert all(o._num_quantile_bins == num_bins for o in valid), (
            "All RunningStats must share the same num_quantile_bins to merge."
        )
        vector_length = valid[0]._mean.size
        assert all(o._mean.size == vector_length for o in valid), (
            "All RunningStats must share the same vector length to merge."
        )

        counts = np.array([o._count for o in valid], dtype=np.float64)
        total_count = counts.sum()
        weights = counts / total_count   # 按样本数加权

        merged_mean = sum(w * o._mean for w, o in zip(weights, valid))                 # 加权均值
        merged_ms = sum(w * o._mean_of_squares for w, o in zip(weights, valid))       # 加权 E[x²]
        merged_min = np.minimum.reduce([o._min for o in valid])                       # 全局最小
        merged_max = np.maximum.reduce([o._max for o in valid])                       # 全局最大

        # Leave a little padding for linspace, consistent with update() init logic (see line 66)
        # 直方图合并:先定全局新分箱,再把每个分片的旧直方图 rebin 到新分箱上累加
        merged_histograms = []
        merged_bin_edges = []
        for dim in range(vector_length):
            new_edges = np.linspace(
                merged_min[dim] - 1e-10, merged_max[dim] + 1e-10, num_bins + 1
            )
            acc = np.zeros(num_bins)
            for o in valid:
                old_edges = o._bin_edges[dim]
                old_hist = o._histograms[dim]
                # Same rebin approach as _adjust_histograms
                rebinned, _ = np.histogram(old_edges[:-1], bins=new_edges, weights=old_hist)
                acc += rebinned
            merged_histograms.append(acc)
            merged_bin_edges.append(new_edges)

        instance = cls()
        instance._num_quantile_bins = num_bins
        instance._count = int(total_count)
        instance._mean = merged_mean
        instance._mean_of_squares = merged_ms
        instance._min = merged_min
        instance._max = merged_max
        instance._histograms = merged_histograms
        instance._bin_edges = merged_bin_edges
        return instance


class _NormStatsDict(pydantic.BaseModel):
    """norm_stats.json 的顶层结构:{ norm_stats: {feature: NormStats}, count: N }。"""
    norm_stats: dict[str, NormStats]
    count: int


def serialize_json(norm_stats: dict[str, NormStats], count: int) -> str:
    """Serialize the running statistics to a JSON string."""
    """把统计量 dict 序列化成 JSON 字符串。"""
    return _NormStatsDict(norm_stats=norm_stats, count=count).model_dump_json(indent=2)


def deserialize_json(data: str) -> dict[str, NormStats]:
    """Deserialize the running statistics from a JSON string."""
    """从 JSON 字符串反序列化出统计量 dict。"""
    return _NormStatsDict(**json.loads(data)).norm_stats


def save(directory: pathlib.Path | str, norm_stats: dict[str, NormStats], count: int) -> None:
    """Save the normalization stats to a directory."""
    """把统计量写成 JSON 文件(注意:参数名 directory,但实际是把文件写到该 path)。"""
    path = pathlib.Path(directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_json(norm_stats, count))


def load(directory: pathlib.Path | str) -> dict[str, NormStats]:
    """Load the normalization stats from a directory."""
    """从目录读取 norm_stats.json 并反序列化(注意:实际读 directory/norm_stats.json)。"""
    path = pathlib.Path(directory) / "norm_stats.json"
    if not path.exists():
        raise FileNotFoundError(f"Norm stats file not found at: {path}")
    return deserialize_json(path.read_text())


class RunningStatsState(pydantic.BaseModel):
    """Model for persisting the internal state of RunningStats"""
    """【注意】此处为重复定义(与本文件上方第 21 行的 RunningStatsState 完全相同,后者覆盖前者)。"""
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    count: int
    mean: numpydantic.NDArray
    mean_of_squares: numpydantic.NDArray
    min_val: numpydantic.NDArray
    max_val: numpydantic.NDArray
    histograms: numpydantic.NDArray  # Shape: (vector_length, num_bins)
    bin_edges: numpydantic.NDArray   # Shape: (vector_length, num_bins + 1)
    num_quantile_bins: int

def save_running_state(path: pathlib.Path | str, stats: dict):
    """Save the full computed intermediate state to JSON"""
    """保存完整的中间状态(可用于断点续算,与只存 NormStats 的 save 不同)。"""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stats = {key: state.get_state().model_dump_json() for key, state in stats.items()}
    json.dumps(stats)
    path.write_text(json.dumps(stats))

def load_running_state(path: pathlib.Path | str) -> RunningStats:
    """Load intermediate state from JSON and restore a RunningStats object"""
    """从 JSON 读回中间状态,还原成 RunningStats dict(断点续算用)。"""
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"State file not found at: {path}")
    data = json.loads(path.read_text())
    stats = {key: RunningStats.from_state(RunningStatsState(**json.loads(state))) for key, state in data.items()}
    return stats
