"""
butt_contact_analyzer.py

V1.1: Pose-only 臀部离凳检测（Butt-Off-Bench），修复 V1 的四个核心问题。

V1 问题修复：
  1. threshold 不再被 MIN_LIFT_PX=15px 绑死 → 双级自适应阈值（suspected/confirmed）
  2. baseline 不再取整个 eccentric → 只取 eccentric 前 25%（接近 setup）
  3. 肩部补偿系数 0.5 太武断 → 改为 0.25
  4. persistence 5 帧过严 → 双级（suspected 3帧 / confirmed 5帧）
  5. 扫描整个 rep → 只分析 concentric
  6. 没有时序平滑 → median filter（window=5）
  7. 日志不清晰 → 输出 peak vs threshold、longest_run、具体原因

核心原则：
  - Arch（正常起桥） ≠ Butt-Off（臀部真正离凳）
  - 只用 Pose 无法 100% 证明臀部离凳，只能给出"与臀部离凳相符的骨盆抬升模式"
  - V1.1 主要输出 SUSPECTED，CONFIRMED 需要 V2（bench plane detection）

检测信号：
  1. pelvis_center = midpoint(left_hip, right_hip)
  2. shoulder_center = midpoint(left_shoulder, right_shoulder)
  3. setup_baseline = median(pelvis_y during eccentric 前 25%)
  4. relative_lift = Δpelvis_y - 0.25 * Δshoulder_y（concentric 段，平滑后）
  5. persistence = 相对抬升超过阈值的最长连续帧段

状态：
  - normal_arch: 正常起桥，相对抬升小或持续不足
  - suspected_lift: 疑似臀部离凳（超过 suspected 阈值且持续 >=3帧）
  - confirmed_lift: 高置信度臀部离凳模式（超过 confirmed 阈值且持续 >=5帧，V1.1 很难达到）
  - insufficient_data: 数据不足
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np

from app.domain.models import RepContext


@dataclass
class LiftRun:
    """一次持续抬升段的详细信息"""
    start: int
    end: int
    duration: int
    max_lift: float
    mean_lift: float


@dataclass
class ButtContactResult:
    """臀部接触检测结果"""
    status: str  # "normal_arch" / "suspected_lift" / "confirmed_lift" / "insufficient_data"
    confidence: float = 0.0
    max_relative_lift: Optional[float] = None  # 最大相对抬升（像素）
    separated_frames: int = 0  # 最长持续抬升帧数
    baseline_pelvis_y: Optional[float] = None
    torso_length: Optional[float] = None
    suspected_threshold: Optional[float] = None
    confirmed_threshold: Optional[float] = None
    reason: str = ""  # 判定原因
    detail: str = ""


class ButtContactAnalyzer:
    """
    V1.1: Pose-only 臀部离凳检测。

    不检测 bench plane，只用骨盆相对肩部的抬升模式判断。
    主要输出 SUSPECTED，CONFIRMED 需要 V2。
    """

    # ==============================
    # V1.1 参数（双级自适应阈值）
    # ==============================

    # 相对躯干长度阈值
    SUSPECTED_LIFT_RATIO = 0.12   # 12% torso length
    CONFIRMED_LIFT_RATIO = 0.20   # 20% torso length

    # 最低像素阈值（兜底，防止 torso 太小时阈值过小）
    MIN_SUSPECTED_PX = 4.0
    MIN_CONFIRMED_PX = 7.0

    # 连续帧要求（双级）
    MIN_SUSPECTED_FRAMES = 3
    MIN_CONFIRMED_FRAMES = 5

    # baseline 区间（eccentric 前 25%）
    BASELINE_RATIO = 0.25

    # 肩部补偿系数（V1 是 0.5，太激进，改为 0.25）
    SHOULDER_REFERENCE_WEIGHT = 0.25

    # 平滑窗口
    SMOOTH_WINDOW = 5

    def analyze(self, rep: RepContext) -> ButtContactResult:
        """分析单个 rep 的臀部接触状态"""

        print(f"\n   🍑 [ButtContact V1.1] Rep {rep.rep_index} 开始分析")

        # 1. 计算骨盆中心和肩部中心
        pelvis_y, shoulder_y, torso_length = self._compute_centers(rep)

        if pelvis_y is None or len(pelvis_y) < 5:
            print(f"   🍑 [ButtContact V1.1] Rep {rep.rep_index}: 缺少髋部坐标数据 → INSUFFICIENT_DATA")
            return ButtContactResult(
                status="insufficient_data",
                confidence=0.0,
                reason="missing_hip_data",
                detail="缺少髋部坐标数据",
            )

        pelvis_valid = int(np.sum(np.isfinite(pelvis_y)))
        shoulder_valid = int(np.sum(np.isfinite(shoulder_y))) if shoulder_y is not None else 0
        print(f"   🍑 [ButtContact V1.1] Rep {rep.rep_index}: "
              f"骨盆有效帧={pelvis_valid}/{len(pelvis_y)}, "
              f"肩部有效帧={shoulder_valid}, "
              f"躯干长度={torso_length:.0f}px" if torso_length else
              f"   🍑 [ButtContact V1.1] Rep {rep.rep_index}: "
              f"骨盆有效帧={pelvis_valid}/{len(pelvis_y)}, "
              f"肩部有效帧={shoulder_valid}, 躯干长度=N/A")

        if shoulder_y is None or len(shoulder_y) < 5:
            shoulder_y = np.full_like(pelvis_y, np.nanmedian(pelvis_y))
            print(f"   🍑 [ButtContact V1.1] Rep {rep.rep_index}: 肩部数据不足，退化为纯骨盆绝对抬升")

        # 2. 计算 setup baseline（eccentric 前 25%，不是整个 eccentric）
        baseline = self._compute_baseline(rep, pelvis_y)

        if baseline is None or not np.isfinite(baseline):
            print(f"   🍑 [ButtContact V1.1] Rep {rep.rep_index}: 无法计算 setup baseline → INSUFFICIENT_DATA")
            return ButtContactResult(
                status="insufficient_data",
                confidence=0.0,
                reason="baseline_computation_failed",
                detail="无法计算 setup baseline",
            )

        print(f"   🍑 [ButtContact V1.1] Rep {rep.rep_index}: setup baseline pelvis_y={baseline:.1f}px (eccentric 前{self.BASELINE_RATIO*100:.0f}%)")

        # 3. 只截取 concentric 段（V1 扫描整个 rep 是错的）
        con_start = max(0, rep.concentric_start - rep.start_frame)
        con_end = min(len(pelvis_y), rep.concentric_end - rep.start_frame + 1)

        if con_end - con_start < 5:
            print(f"   🍑 [ButtContact V1.1] Rep {rep.rep_index}: concentric 段过短 ({con_end-con_start}帧) → INSUFFICIENT_DATA")
            return ButtContactResult(
                status="insufficient_data",
                confidence=0.0,
                reason="concentric_too_short",
                detail=f"concentric 段过短 ({con_end-con_start}帧)",
            )

        pelvis_con = pelvis_y[con_start:con_end].copy()
        shoulder_con = shoulder_y[con_start:con_end].copy()
        print(f"   🍑 [ButtContact V1.1] Rep {rep.rep_index}: concentric 段 [{con_start}:{con_end}], 共{len(pelvis_con)}帧")

        # 4. median smoothing（V1 没有平滑，jitter 导致尖锐峰值）
        pelvis_con = self._smooth(pelvis_con, self.SMOOTH_WINDOW)
        shoulder_con = self._smooth(shoulder_con, self.SMOOTH_WINDOW)

        # 5. 计算 relative_lift（concentric 段，平滑后）
        # 图像坐标 y 向下为正，骨盆向上 = pelvis_y 减小 = baseline - pelvis_y > 0
        shoulder_baseline = float(np.nanmedian(shoulder_con[:max(1, len(shoulder_con) // 3)]))

        pelvis_lift = baseline - pelvis_con  # 向上为正
        shoulder_lift = shoulder_baseline - shoulder_con  # 向上为正

        # 减去肩部运动参考（系数 0.25，比 V1 的 0.5 保守）
        relative_lift = pelvis_lift - self.SHOULDER_REFERENCE_WEIGHT * shoulder_lift

        max_pelvis_lift = float(np.nanmax(pelvis_lift))
        max_shoulder_lift = float(np.nanmax(shoulder_lift))
        max_relative_lift_raw = float(np.nanmax(relative_lift))
        print(f"   🍑 [ButtContact V1.1] Rep {rep.rep_index}: "
              f"max_pelvis_lift={max_pelvis_lift:.1f}px, "
              f"max_shoulder_lift={max_shoulder_lift:.1f}px, "
              f"max_relative_lift={max_relative_lift_raw:.1f}px "
              f"(补偿系数={self.SHOULDER_REFERENCE_WEIGHT})")

        # 6. 计算双级自适应阈值
        if torso_length is not None and torso_length > 0:
            suspected_threshold = max(self.MIN_SUSPECTED_PX, self.SUSPECTED_LIFT_RATIO * torso_length)
            confirmed_threshold = max(self.MIN_CONFIRMED_PX, self.CONFIRMED_LIFT_RATIO * torso_length)
        else:
            suspected_threshold = self.MIN_SUSPECTED_PX
            confirmed_threshold = self.MIN_CONFIRMED_PX

        print(f"   🍑 [ButtContact V1.1] Rep {rep.rep_index}: "
              f"suspected_threshold={suspected_threshold:.1f}px "
              f"({self.SUSPECTED_LIFT_RATIO*100:.0f}% torso={torso_length:.0f}px, min={self.MIN_SUSPECTED_PX}px), "
              f"confirmed_threshold={confirmed_threshold:.1f}px "
              f"({self.CONFIRMED_LIFT_RATIO*100:.0f}% torso, min={self.MIN_CONFIRMED_PX}px)")

        # 7. 检测持续抬升（双级）
        suspected_run = self._find_longest_run(relative_lift, suspected_threshold)
        confirmed_run = self._find_longest_run(relative_lift, confirmed_threshold)

        print(f"   🍑 [ButtContact V1.1] Rep {rep.rep_index}: "
              f"suspected: longest_run={suspected_run.duration if suspected_run else 0}帧 "
              f"(min={self.MIN_SUSPECTED_FRAMES}), "
              f"max={suspected_run.max_lift if suspected_run else 0:.1f}px")
        print(f"   🍑 [ButtContact V1.1] Rep {rep.rep_index}: "
              f"confirmed: longest_run={confirmed_run.duration if confirmed_run else 0}帧 "
              f"(min={self.MIN_CONFIRMED_FRAMES}), "
              f"max={confirmed_run.max_lift if confirmed_run else 0:.1f}px")

        # 8. 判断状态
        # CONFIRMED: 超过 confirmed 阈值且持续 >= confirmed 帧数
        if (confirmed_run is not None
                and confirmed_run.duration >= self.MIN_CONFIRMED_FRAMES):
            status = "confirmed_lift"
            confidence = min(0.85, 0.6 + 0.25 * min(1.0, confirmed_run.duration / (self.MIN_CONFIRMED_FRAMES * 2)))
            reason = "confirmed_threshold_exceeded_with_persistence"
            max_lift = confirmed_run.max_lift
            separated_frames = confirmed_run.duration
            status_label = "CONFIRMED_LIFT (高置信度臀部离凳模式)"

        # SUSPECTED: 超过 suspected 阈值且持续 >= suspected 帧数
        elif (suspected_run is not None
                and suspected_run.duration >= self.MIN_SUSPECTED_FRAMES):
            status = "suspected_lift"
            lift_ratio = min(1.0, suspected_run.max_lift / (suspected_threshold * 2.0))
            persistence_ratio = min(1.0, suspected_run.duration / (self.MIN_SUSPECTED_FRAMES * 2.0))
            confidence = 0.4 + 0.3 * lift_ratio + 0.3 * persistence_ratio
            confidence = min(0.79, confidence)  # suspected 最高 0.79，confirmed 从 0.80 开始
            reason = "suspected_threshold_exceeded_with_persistence"
            max_lift = suspected_run.max_lift
            separated_frames = suspected_run.duration
            status_label = "SUSPECTED_LIFT (疑似臀部离凳)"

        # 峰值超过 suspected 但持续不足 → NORMAL_ARCH（但记录原因）
        elif max_relative_lift_raw >= suspected_threshold:
            status = "normal_arch"
            confidence = 0.65
            reason = "peak_above_threshold_but_insufficient_persistence"
            max_lift = max_relative_lift_raw
            separated_frames = suspected_run.duration if suspected_run else 0
            status_label = "NORMAL_ARCH (峰值超阈值但持续不足)"

        # 峰值低于 suspected 阈值 → NORMAL_ARCH
        else:
            status = "normal_arch"
            confidence = 0.75
            reason = "peak_below_threshold"
            max_lift = max_relative_lift_raw
            separated_frames = 0
            status_label = "NORMAL_ARCH (正常起桥)"

        print(f"   🍑 [ButtContact V1.1] Rep {rep.rep_index}: → {status_label}, "
              f"confidence={confidence:.2f}, reason={reason}")

        return ButtContactResult(
            status=status,
            confidence=round(confidence, 2),
            max_relative_lift=round(float(max_lift), 1),
            separated_frames=separated_frames,
            baseline_pelvis_y=round(float(baseline), 1),
            torso_length=round(float(torso_length), 1) if torso_length else None,
            suspected_threshold=round(float(suspected_threshold), 1),
            confirmed_threshold=round(float(confirmed_threshold), 1),
            reason=reason,
            detail=(
                f"max_lift={max_lift:.1f}px "
                f"suspected_thr={suspected_threshold:.1f}px "
                f"confirmed_thr={confirmed_threshold:.1f}px "
                f"persistence={separated_frames}frames "
                f"torso_len={torso_length:.0f}px"
            ),
        )

    # =========================================================================
    # 内部方法
    # =========================================================================

    def _compute_centers(
        self, rep: RepContext
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
        """
        计算骨盆中心和肩部中心的 y 坐标序列，以及躯干长度参考。
        """
        pelvis_y = self._midpoint_y(rep.left_hip, rep.right_hip)
        shoulder_y = self._midpoint_y(rep.left_shoulder, rep.right_shoulder)

        torso_length = None
        if pelvis_y is not None and shoulder_y is not None:
            valid = np.isfinite(pelvis_y) & np.isfinite(shoulder_y)
            if valid.sum() >= 5:
                torso_length = float(np.nanmedian(np.abs(pelvis_y[valid] - shoulder_y[valid])))

        return pelvis_y, shoulder_y, torso_length

    @staticmethod
    def _midpoint_y(
        left: Optional[np.ndarray],
        right: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """计算左右侧坐标的 y 中点序列"""
        if left is None and right is None:
            return None

        if left is not None and right is not None:
            n = min(len(left), len(right))
            ly = left[:n, 1] if left.ndim == 2 else left[:n]
            ry = right[:n, 1] if right.ndim == 2 else right[:n]
            mid = np.full(n, np.nan, dtype=np.float64)
            both = np.isfinite(ly) & np.isfinite(ry)
            mid[both] = (ly[both] + ry[both]) / 2.0
            left_only = np.isfinite(ly) & ~np.isfinite(ry)
            mid[left_only] = ly[left_only]
            right_only = ~np.isfinite(ly) & np.isfinite(ry)
            mid[right_only] = ry[right_only]
            return mid

        single = left if left is not None else right
        if single.ndim == 2:
            return single[:, 1].copy()
        return single.copy()

    def _compute_baseline(self, rep: RepContext, pelvis_y: np.ndarray) -> Optional[float]:
        """
        V1.1: setup baseline = eccentric 前 25% 的骨盆位置中位数。
        （V1 取整个 eccentric，会被运动过程中的骨盆变化污染）
        """
        if pelvis_y is None or len(pelvis_y) < 5:
            return None

        ecc_start = max(0, rep.eccentric_start - rep.start_frame)
        ecc_end = min(len(pelvis_y), rep.eccentric_end - rep.start_frame + 1)

        if ecc_end - ecc_start < 5:
            # 退化：用前 1/3
            ecc_end = max(ecc_start + 1, len(pelvis_y) // 3)

        seg = pelvis_y[ecc_start:ecc_end]
        seg = seg[np.isfinite(seg)]

        if len(seg) < 3:
            return None

        # 只使用 eccentric 前 BASELINE_RATIO（25%）
        n = max(3, int(len(seg) * self.BASELINE_RATIO))
        baseline_values = seg[:n]

        return float(np.median(baseline_values))

    @staticmethod
    def _smooth(x: np.ndarray, window: int = 5) -> np.ndarray:
        """
        Median filter（中值滤波），抗 Pose jitter。
        比 mean filter 更好地保留边缘，同时去除尖锐噪声。
        """
        if len(x) < window:
            return x.copy()

        out = x.copy()
        half = window // 2

        for i in range(len(x)):
            lo = max(0, i - half)
            hi = min(len(x), i + half + 1)
            vals = x[lo:hi]
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                out[i] = np.median(vals)

        return out

    @staticmethod
    def _find_longest_run(series: np.ndarray, threshold: float) -> Optional[LiftRun]:
        """
        找到超过阈值的最长连续段，返回详细信息。

        Returns:
            LiftRun 或 None（没有超过阈值的段）
        """
        if series is None or len(series) == 0:
            return None

        above = series > threshold
        if not above.any():
            return None

        # 找所有连续段
        runs = []
        start = None

        for i, flag in enumerate(above):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                runs.append((start, i - 1))
                start = None

        if start is not None:
            runs.append((start, len(above) - 1))

        if not runs:
            return None

        # 取最长的段
        best_start, best_end = max(runs, key=lambda x: x[1] - x[0] + 1)
        segment = series[best_start:best_end + 1]
        segment = segment[np.isfinite(segment)]

        if len(segment) == 0:
            return None

        return LiftRun(
            start=best_start,
            end=best_end,
            duration=best_end - best_start + 1,
            max_lift=float(np.nanmax(segment)),
            mean_lift=float(np.nanmean(segment)),
        )
