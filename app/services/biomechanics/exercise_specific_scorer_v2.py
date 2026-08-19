"""
exercise_specific_scorer_v2.py (v2)

修正：
  - bar_path 严格切 concentric 段
  - 缺数据 → score=None, status=INSUFFICIENT_DATA
  - 动态权重归一化
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import numpy as np
import warnings

from app.domain.models import RepContext
from app.domain.enums import ValidationStatus, MetricStatus


@dataclass
class MetricResult:
    key: str
    raw: Optional[float] = None
    score: Optional[float] = None
    status: MetricStatus = MetricStatus.VALID
    detail: str = ""


@dataclass
class LayerResult:
    layer_name: str
    score: Optional[float] = None
    status: MetricStatus = MetricStatus.VALID
    metrics: List[MetricResult] = field(default_factory=list)
    detail: str = ""


@dataclass
class RepScoreResult:
    rep_index: int
    layers: Dict[str, LayerResult] = field(default_factory=dict)
    technique_score: Optional[float] = None
    movement_quality: Optional[float] = None
    safety_score: Optional[float] = None
    performance_score: Optional[float] = None
    overall_score: Optional[float] = None
    status: MetricStatus = MetricStatus.VALID


TECHNIQUE_WEIGHTS: Dict[str, float] = {
    "bar_path": 0.25,
    "elbow_tuck": 0.20,
    "touch_point": 0.15,
    "symmetry": 0.20,
    "tempo": 0.20,
}

LAYER_WEIGHTS: Dict[str, float] = {
    "technique": 0.40,
    "movement_quality": 0.25,
    "safety": 0.20,
    "performance": 0.15,
}


# ═══════════════════════════════════════
#  指标计算函数
# ═══════════════════════════════════════

def compute_bar_path(rep: RepContext) -> MetricResult:
    if not rep.has_concentric:
        return MetricResult("bar_path", None, None,
                            MetricStatus.INSUFFICIENT_DATA, "无 concentric 阶段")
    if rep.bilateral_elbow is None:
        return MetricResult("bar_path", None, None,
                            MetricStatus.INSUFFICIENT_DATA, "无肘部数据")

    cs = rep.concentric_start - rep.start_frame
    ce = rep.concentric_end - rep.start_frame
    ce = min(ce, len(rep.bilateral_elbow))

    if ce - cs < 3:
        return MetricResult("bar_path", None, None,
                            MetricStatus.INSUFFICIENT_DATA, "concentric 帧数不足")

    con_segment = rep.bilateral_elbow[cs: ce]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        std_val = float(np.nanstd(con_segment))
    score = max(0.0, 100.0 - std_val * 5.0)
    return MetricResult("bar_path", raw=std_val, score=score)


def compute_elbow_tuck(rep: RepContext) -> MetricResult:
    bottom_phase = rep.get_phase("bottom")
    if bottom_phase is None or rep.bilateral_elbow is None:
        return MetricResult("elbow_tuck", None, None, MetricStatus.INSUFFICIENT_DATA)

    bs = bottom_phase.start_frame - rep.start_frame
    be = min(bottom_phase.end_frame - rep.start_frame, len(rep.bilateral_elbow))
    if be - bs < 2:
        return MetricResult("elbow_tuck", None, None, MetricStatus.INSUFFICIENT_DATA)

    seg = rep.bilateral_elbow[bs: be]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        min_angle = float(np.nanmin(seg))
    ideal = 80.0
    score = max(0.0, 100.0 - abs(min_angle - ideal) * 2.0)
    return MetricResult("elbow_tuck", raw=min_angle, score=score)


def compute_touch_point(rep: RepContext) -> MetricResult:
    bottom_phase = rep.get_phase("bottom")
    if bottom_phase is None or rep.bilateral_elbow is None:
        return MetricResult("touch_point", None, None, MetricStatus.INSUFFICIENT_DATA)

    bs = bottom_phase.start_frame - rep.start_frame
    be = min(bottom_phase.end_frame - rep.start_frame, len(rep.bilateral_elbow))
    if be - bs < 2:
        return MetricResult("touch_point", None, None, MetricStatus.INSUFFICIENT_DATA)

    seg = rep.bilateral_elbow[bs: be]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        std_val = float(np.nanstd(seg))
    score = max(0.0, 100.0 - std_val * 8.0)
    return MetricResult("touch_point", raw=std_val, score=score)


def compute_symmetry(rep: RepContext) -> MetricResult:
    if rep.left_elbow is None or rep.right_elbow is None:
        return MetricResult("symmetry", None, None,
                            MetricStatus.INSUFFICIENT_DATA, "缺少单侧数据")

    cs = max(0, rep.concentric_start - rep.start_frame)
    ce = min(len(rep.left_elbow), rep.concentric_end - rep.start_frame)
    if ce - cs < 2:
        return MetricResult("symmetry", None, None, MetricStatus.INSUFFICIENT_DATA)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lm = float(np.nanmean(rep.left_elbow[cs: ce]))
        rm = float(np.nanmean(rep.right_elbow[cs: ce]))
    if not np.isfinite(lm) or not np.isfinite(rm):
        return MetricResult("symmetry", None, None,
                            MetricStatus.INSUFFICIENT_DATA, "单侧数据全为 NaN")
    asym = abs(lm - rm)
    score = max(0.0, 100.0 - asym * 5.0)
    return MetricResult("symmetry", raw=asym, score=score)


def compute_tempo(rep: RepContext) -> MetricResult:
    if rep.eccentric_duration <= 0 or rep.concentric_duration <= 0:
        return MetricResult("tempo", None, None, MetricStatus.INSUFFICIENT_DATA)

    ratio = rep.eccentric_duration / rep.concentric_duration
    if 1.2 <= ratio <= 3.0:
        score = 100.0
    elif ratio < 1.2:
        score = max(0.0, 100.0 - (1.2 - ratio) * 40.0)
    else:
        score = max(0.0, 100.0 - (ratio - 3.0) * 20.0)
    return MetricResult("tempo", raw=ratio, score=score)


# ═══════════════════════════════════════
#  聚合
# ═══════════════════════════════════════

def aggregate_layer(
    layer_name: str,
    metrics: List[MetricResult],
    weights: Dict[str, float],
) -> LayerResult:
    valid = [(m, weights.get(m.key, 0.0))
             for m in metrics
             if m.status == MetricStatus.VALID and m.score is not None]

    if not valid:
        return LayerResult(layer_name, None, MetricStatus.INSUFFICIENT_DATA,
                           metrics, "所有指标数据不足")

    tw = sum(w for _, w in valid)
    if tw <= 0:
        return LayerResult(layer_name, None, MetricStatus.INSUFFICIENT_DATA, metrics)

    weighted = sum(m.score * (w / tw) for m, w in valid)
    return LayerResult(layer_name, round(weighted, 1), MetricStatus.VALID,
                       metrics, f"{len(valid)}/{len(metrics)} 指标有效")


# ═══════════════════════════════════════
#  主 Scorer
# ═══════════════════════════════════════

class ExerciseSpecificScorerV2:

    def score_rep(self, rep: RepContext) -> RepScoreResult:
        r = RepScoreResult(rep_index=rep.rep_index)

        if rep.validation_status != ValidationStatus.VALID:
            r.status = MetricStatus.INSUFFICIENT_DATA
            r.overall_score = None
            return r

        tech = [compute_bar_path(rep), compute_elbow_tuck(rep),
                compute_touch_point(rep), compute_symmetry(rep), compute_tempo(rep)]
        tl = aggregate_layer("technique", tech, TECHNIQUE_WEIGHTS)
        r.layers["technique"] = tl
        r.technique_score = tl.score

        mq = self._mq_metrics(rep)
        ml = aggregate_layer("movement_quality", mq, {"rom_consistency": .5, "velocity_smoothness": .5})
        r.layers["movement_quality"] = ml
        r.movement_quality = ml.score

        sf = self._safety_metrics(rep)
        sl = aggregate_layer("safety", sf, {"joint_stress": .5, "control": .5})
        r.layers["safety"] = sl
        r.safety_score = sl.score

        pf = self._perf_metrics(rep)
        pl = aggregate_layer("performance", pf, {"power": .5, "rom": .5})
        r.layers["performance"] = pl
        r.performance_score = pl.score

        vl = [(n, w) for n, w in LAYER_WEIGHTS.items() if r.layers[n].score is not None]
        if vl:
            tw = sum(w for _, w in vl)
            r.overall_score = round(sum(r.layers[n].score * (w / tw) for n, w in vl) / 10.0, 1)
        else:
            r.overall_score = None
            r.status = MetricStatus.INSUFFICIENT_DATA

        return r

    def _mq_metrics(self, rep: RepContext) -> List[MetricResult]:
        ms = []
        if rep.actual_rom > 0:
            dev = abs(rep.actual_rom - 80.0)
            ms.append(MetricResult("rom_consistency", rep.actual_rom, max(0, 100 - dev * 1.5)))
        else:
            ms.append(MetricResult("rom_consistency", None, None, MetricStatus.INSUFFICIENT_DATA))

        if rep.concentric_velocity is not None and len(rep.concentric_velocity) > 3:
            jerk_std = float(np.std(np.diff(rep.concentric_velocity)))
            ms.append(MetricResult("velocity_smoothness", jerk_std, max(0, 100 - jerk_std * 2)))
        else:
            ms.append(MetricResult("velocity_smoothness", None, None, MetricStatus.INSUFFICIENT_DATA))
        return ms

    def _safety_metrics(self, rep: RepContext) -> List[MetricResult]:
        ms = []
        if rep.bottom_angle > 0:
            s = max(0, 100 - max(0, 50 - rep.bottom_angle) * 3) if rep.bottom_angle < 50 else 100.0
            ms.append(MetricResult("joint_stress", rep.bottom_angle, s))
        else:
            ms.append(MetricResult("joint_stress", None, None, MetricStatus.INSUFFICIENT_DATA))

        if rep.peak_eccentric_velocity is not None:
            s = max(0, 100 - max(0, rep.peak_eccentric_velocity - 300) * 0.5)
            ms.append(MetricResult("control", rep.peak_eccentric_velocity, s))
        else:
            ms.append(MetricResult("control", None, None, MetricStatus.INSUFFICIENT_DATA))
        return ms

    def _perf_metrics(self, rep: RepContext) -> List[MetricResult]:
        ms = []
        if rep.peak_concentric_velocity is not None:
            ms.append(MetricResult("power", rep.peak_concentric_velocity,
                                   min(100, rep.peak_concentric_velocity / 2.0)))
        else:
            ms.append(MetricResult("power", None, None, MetricStatus.INSUFFICIENT_DATA))

        if rep.actual_rom > 0:
            ms.append(MetricResult("rom", rep.actual_rom, min(100, rep.actual_rom)))
        else:
            ms.append(MetricResult("rom", None, None, MetricStatus.INSUFFICIENT_DATA))
        return ms
    
    def format_v2_results_for_frontend(
        rep_scores: list,
        set_errors: list,
        fatigue_result=None,
        exercise_type: str = "unknown",
    ) -> dict:
        """
        将 V2 引擎原始输出转为前端可直接消费的 JSON 结构。
        """
        # 1. Rep 级别
        reps_out = []
        for rs in rep_scores:
            reps_out.append({
                "rep_index": rs.rep_index,
                "score": rs.total_score,
                "grade": rs.grade,
                "errors": [
                    {
                        "code": e.code,
                        "severity": e.severity,
                        "message": e.message,
                        "deduction": e.deduction,
                    }
                    for e in (rs.errors or [])
                ],
                "metrics": {
                    "rom": rs.metrics.get("rom"),
                    "tempo_ratio": rs.metrics.get("tempo_ratio"),
                    "peak_concentric_velocity": rs.metrics.get("peak_concentric_velocity"),
                    "bottom_dwell_time": rs.metrics.get("bottom_dwell_time"),
                },
            })

        # 2. Set 级别错误
        set_errors_out = [
            {
                "code": e.code,
                "severity": e.severity,
                "message": e.message,
            }
            for e in (set_errors or [])
        ]

        # 3. 疲劳
        fatigue_out = None
        if fatigue_result and getattr(fatigue_result, "status", "") == "valid":
            fatigue_out = {
                "velocity_loss_pct": fatigue_result.velocity_loss_pct,
                "fatigue_level": fatigue_result.fatigue_level,
                "estimated_rir": fatigue_result.estimated_rir,
                "trend": fatigue_result.trend,
            }

        # 4. 汇总
        valid_scores = [r["score"] for r in reps_out if r["score"] is not None]
        avg_score = round(sum(valid_scores) / len(valid_scores), 1) if valid_scores else 0.0

        return {
            "exercise_type": exercise_type,
            "total_reps": len(reps_out),
            "average_score": avg_score,
            "reps": reps_out,
            "set_errors": set_errors_out,
            "fatigue": fatigue_out,
        }

def format_v2_results_for_frontend(rep_scores, set_errors, fatigue_result=None, exercise_type='unknown'):
    reps_out = []
    for rs in rep_scores:
        reps_out.append({
            'rep_index': rs.rep_index,
            'score': rs.total_score,
            'grade': rs.grade,
            'errors': [{'code': e.code, 'severity': e.severity, 'message': e.message, 'deduction': e.deduction} for e in (rs.errors or [])],
            'metrics': {k: rs.metrics.get(k) for k in ['rom','tempo_ratio','peak_concentric_velocity','bottom_dwell_time']},
        })
    set_errors_out = [{'code': e.code, 'severity': e.severity, 'message': e.message} for e in (set_errors or [])]
    fatigue_out = None
    if fatigue_result and getattr(fatigue_result, 'status', '') == 'valid':
        fatigue_out = {'velocity_loss_pct': fatigue_result.velocity_loss_pct, 'fatigue_level': fatigue_result.fatigue_level, 'estimated_rir': fatigue_result.estimated_rir, 'trend': fatigue_result.trend}
    valid_scores = [r['score'] for r in reps_out if r['score'] is not None]
    avg_score = round(sum(valid_scores)/len(valid_scores),1) if valid_scores else 0.0
    return {'exercise_type': exercise_type, 'total_reps': len(reps_out), 'average_score': avg_score, 'reps': reps_out, 'set_errors': set_errors_out, 'fatigue': fatigue_out}
