from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np
from app.domain.enums import ValidationStatus, SignalSource


@dataclass
class PhaseSegment:
    """Rep 内的一个阶段切片"""
    name: str                # eccentric / bottom / concentric / lockout
    start_frame: int         # 绝对帧号
    end_frame: int           # 绝对帧号
    start_time: float
    end_time: float
    peak_velocity: Optional[float] = None
    mean_velocity: Optional[float] = None


@dataclass
class RepContext:
    """
    统一 Rep 数据契约。
    CycleRepDetector → PhaseBuilder → RepContext → 下游全部消费此结构。
    """
    rep_index: int

    # ── CycleRepDetector 事件锚点（绝对帧号） ──
    start_frame: int         # TOP1
    bottom_frame: int        # BOTTOM
    end_frame: int           # TOP2

    # ── 事件锚点角度 ──
    top_angle: float         # TOP1 处角度
    bottom_angle: float      # BOTTOM 处角度
    top2_angle: float        # TOP2 处角度

    # ── interval 真实 ROM（actual max − min） ──
    actual_max_angle: float
    actual_min_angle: float
    actual_rom: float

    # ── 时间 ──
    start_time: float
    bottom_time: float
    end_time: float
    total_duration: float

    # ── 阶段边界（绝对帧号，PhaseBuilder 填充） ──
    eccentric_start: int = -1
    eccentric_end: int = -1
    concentric_start: int = -1
    concentric_end: int = -1
    bottom_zone_start: int = -1
    bottom_zone_end: int = -1

    # ── 阶段时长 ──
    eccentric_duration: float = 0.0
    concentric_duration: float = 0.0
    bottom_dwell_time: float = 0.0

    # ── 速度（度/秒，PhaseBuilder 填充） ──
    eccentric_velocity: Optional[np.ndarray] = None
    concentric_velocity: Optional[np.ndarray] = None
    peak_eccentric_velocity: Optional[float] = None
    peak_concentric_velocity: Optional[float] = None
    mean_concentric_velocity: Optional[float] = None

    # ── bottom 附近动力学（bounce 检测用） ──
    pre_bottom_velocity: Optional[float] = None
    bottom_acceleration: Optional[float] = None
    direction_reversal_frames: int = 0

    # ── Rep-relative 双侧信号 ──
    left_elbow: Optional[np.ndarray] = None
    right_elbow: Optional[np.ndarray] = None
    bilateral_elbow: Optional[np.ndarray] = None
    bilateral_valid_ratio: float = 0.0
    signal_source: SignalSource = SignalSource.BILATERAL

    # ── Rep-relative wrist 坐标（形状 (N, 2)，用于 bar path 检测）──
    left_wrist: Optional[np.ndarray] = None
    right_wrist: Optional[np.ndarray] = None

    # ── 阶段列表 ──
    phases: List[PhaseSegment] = field(default_factory=list)

    # ── 验证 ──
    validation_status: ValidationStatus = ValidationStatus.VALID
    fps: float = 30.0

    # ── 便捷属性 ──
    @property
    def duration(self) -> float:
        return self.total_duration

    @property
    def has_phases(self) -> bool:
        return len(self.phases) > 0

    @property
    def has_concentric(self) -> bool:
        return self.concentric_start >= 0 and self.concentric_end > self.concentric_start

    @property
    def has_eccentric(self) -> bool:
        return self.eccentric_start >= 0 and self.eccentric_end > self.eccentric_start

    def get_phase(self, name: str) -> Optional[PhaseSegment]:
        for p in self.phases:
            if p.name == name:
                return p
        return None

    def to_dict(self) -> dict:
        return {
            "rep_index": self.rep_index,
            "frames": [self.start_frame, self.end_frame],
            "bottom_frame": self.bottom_frame,
            "top_angle": round(self.top_angle, 1),
            "bottom_angle": round(self.bottom_angle, 1),
            "actual_rom": round(self.actual_rom, 1),
            "duration": round(self.total_duration, 2),
            "eccentric_duration": round(self.eccentric_duration, 2),
            "concentric_duration": round(self.concentric_duration, 2),
            "bottom_dwell_time": round(self.bottom_dwell_time, 3),
            "peak_concentric_velocity": round(self.peak_concentric_velocity, 1) if self.peak_concentric_velocity else None,
            "mean_concentric_velocity": round(self.mean_concentric_velocity, 1) if self.mean_concentric_velocity else None,
            "signal_source": self.signal_source.value,
            "bilateral_valid_ratio": round(self.bilateral_valid_ratio, 2),
            "validation_status": self.validation_status.value,
            "phases": [{"name": p.name, "frames": [p.start_frame, p.end_frame]} for p in self.phases],
        }

@dataclass
class FramePoseData:
    frame_idx: int
    landmarks: Dict[int, Any]
    angles: Dict[str, float]
    positions: Dict[str, List[float]]
    barbell_pos: Optional[Tuple[int, int]] = None
    timestamp_sec: float = 0.0

@dataclass
class RepCycle:
    rep_index: int
    start_frame: int
    end_frame: int
    phases: Dict[str, int]
    phase_frames: Dict[str, List[int]]
    metrics: Dict[str, Any]
    errors: List[Dict[str, Any]]
    normalized_angles: Dict[str, List[float]]