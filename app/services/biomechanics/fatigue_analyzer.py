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


class FatigueAnalyzerFixed:

    def __init__(self, min_reps: int = 3, baseline_reps: int = 3,
                 high_thresh: float = 0.25, moderate_thresh: float = 0.15):
        self.min_reps = min_reps
        self.baseline_reps = baseline_reps
        self.high_thresh = high_thresh
        self.moderate_thresh = moderate_thresh

    def analyze(self, reps: List[RepContext]) -> FatigueResult:
        valid = [r for r in reps
                 if r.validation_status == ValidationStatus.VALID
                 and r.mean_concentric_velocity is not None
                 and r.mean_concentric_velocity > 0]

        if len(valid) < self.min_reps:
            return FatigueResult(
                status="insufficient_data",
                detail=f"有效 Rep {len(valid)} < {self.min_reps}",
                valid_reps_used=len(valid),
            )

        vels = [r.mean_concentric_velocity for r in valid]
        nb = min(self.baseline_reps, len(vels) - 1)
        base = float(np.mean(vels[:nb]))
        if base <= 0:
            return FatigueResult(status="insufficient_data",
                                 detail="基线速度为零", valid_reps_used=len(valid))

        last = vels[-1]
        loss = (base - last) / base

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

        return FatigueResult(
            velocity_loss_pct=round(loss * 100, 1),
            fatigue_level=level,
            estimated_rir=rir,
            trend=trend,
            valid_reps_used=len(valid),
            status="valid",
            detail=f"base={base:.1f}, last={last:.1f} °/s",
        )