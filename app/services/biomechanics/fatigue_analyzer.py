"""
fatigue_analyzer.py (v2)
基于向心速度的疲劳分析。
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from app.domain.models import RepContext
from app.domain.enums import ValidationStatus


@dataclass
class FatigueResult:
    velocity_loss_pct: float = 0.0
    fatigue_level: str = "unknown"
    estimated_rir: Optional[int] = None
    trend: str = "unknown"
    valid_reps_used: int = 0
    status: str = "valid"
    detail: str = ""
    confidence: float = 0.0
    rir_basis: str = ""


class FatigueAnalyzerFixed:

    def __init__(self, min_reps: int = 3, baseline_reps: int = 3,
                 high_thresh: float = 0.25, moderate_thresh: float = 0.15):
        self.min_reps = min_reps
        self.baseline_reps = baseline_reps
        self.high_thresh = high_thresh
        self.moderate_thresh = moderate_thresh

    def analyze(self, reps: List[RepContext]) -> FatigueResult:
        # P0修复：改用 peak_concentric_velocity（峰值向心速度），
        # 避免 mean_concentric_velocity 被底部停留帧拉低导致负速度损失
        valid = [r for r in reps
                 if r.validation_status == ValidationStatus.VALID
                 and r.peak_concentric_velocity is not None
                 and r.peak_concentric_velocity > 0]

        if len(valid) < self.min_reps:
            return FatigueResult(
                status="insufficient_data",
                detail=f"有效 Rep {len(valid)} < {self.min_reps}",
                valid_reps_used=len(valid),
            )

        vels = [r.peak_concentric_velocity for r in valid]
        nb = min(self.baseline_reps, len(vels) - 1)
        base = float(np.mean(vels[:nb]))
        if base <= 0:
            return FatigueResult(status="insufficient_data",
                                 detail="基线速度为零", valid_reps_used=len(valid))

        last = vels[-1]
        loss = (base - last) / base

        # 负速度损失（末段速度上升）不再直接给 RIR=5，
        # 标记为 velocity_increase，降低置信度，不输出疲劳等级和 RIR
        if loss < -0.15:
            return FatigueResult(
                velocity_loss_pct=round(loss * 100, 1),
                fatigue_level="unclear",
                estimated_rir=None,
                trend="unknown",
                valid_reps_used=len(valid),
                status="velocity_increase",
                confidence=0.4,
                rir_basis="velocity_increase",
                detail=f"末段速度上升 {abs(loss)*100:.1f}%（基线={base:.1f}，末次={last:.1f} °/s），"
                       f"可能是基线不稳定、底部停留变化或动作幅度变化，无法可靠评估疲劳",
            )

        level = ("high" if loss >= self.high_thresh else
                 "moderate" if loss >= self.moderate_thresh else "low")

        if loss <= 0:
            rir = 5
        elif loss < 0.40:
            rir = max(0, int(5 - (loss / 0.40) * 5))
        else:
            rir = 0

        trend = "unknown"
        if len(vels) >= 3:
            slope = np.polyfit(range(len(vels)), vels, 1)[0]
            trend = "declining" if slope < -0.5 else "stable"

        # 置信度：基于有效 Rep 数量和速度变异系数
        cv = float(np.std(vels) / np.mean(vels)) if np.mean(vels) > 0 else 1.0
        confidence = max(0.3, min(0.9, 0.5 + (len(valid) - 3) * 0.08 - cv * 0.3))

        return FatigueResult(
            velocity_loss_pct=round(loss * 100, 1),
            fatigue_level=level,
            estimated_rir=rir,
            trend=trend,
            valid_reps_used=len(valid),
            status="valid",
            confidence=round(confidence, 2),
            rir_basis="velocity_loss",
            detail=f"base={base:.1f}, last={last:.1f} °/s, cv={cv:.2f}",
        )