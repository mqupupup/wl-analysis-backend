# backend/app/services/biomechanics/movement_classification.py
"""
动作分类层：V26 Hybrid Classifier（含三重卧推保护）
"""

from typing import Dict, Tuple, Any
from enum import Enum


class MovementPattern(str, Enum):
    LOWER_BODY_SQUAT = "lower_body_squat"
    LOWER_BODY_HINGE = "lower_body_hinge"
    UPPER_BODY_HORIZONTAL_PUSH = "upper_body_horizontal_push"
    UPPER_BODY_VERTICAL_PUSH = "upper_body_vertical_push"
    UNKNOWN = "unknown"

PATTERN_TO_EXERCISE = {
    MovementPattern.LOWER_BODY_SQUAT: "Squat",
    MovementPattern.LOWER_BODY_HINGE: "Deadlift",
    MovementPattern.UPPER_BODY_HORIZONTAL_PUSH: "Bench Press",
    MovementPattern.UPPER_BODY_VERTICAL_PUSH: "Overhead Press",
}


class V26HybridClassifier:
    """
    V26: 彻底封堵卧推误判

    核心改动：
    1. Stage1: 增加"肩髋共线+极低身体起伏"铁定 lying 规则
    2. Stage2: 增加卧推硬分流（bench_signal）
    3. Stage3_upper: 锁死卧推 + 手腕水平位移区分 Bench/OHP
    4. Stage3_lower: 保险丝加粗
    """

    def classify(self, features: Dict[str, float]) -> Tuple[str, float, Dict[str, Any]]:
        debug = {}

        stage1 = self._stage1_posture_gate(features)
        debug['stage1'] = stage1

        stage2 = self._stage2_motion_chain(features, stage1)
        debug['stage2'] = stage2

        if stage2['is_upper']:
            stage3 = self._stage3_upper(features, stage1, stage2)
        else:
            stage3 = self._stage3_lower(features, stage1, stage2)

        debug['stage3'] = stage3
        exercise = stage3['exercise']

        s1c = stage1.get('confidence', 80)
        s2c = stage2.get('confidence', 80)
        s3c = stage3.get('confidence', 80)
        final_conf = s1c * 0.2 + s2c * 0.3 + s3c * 0.5

        debug['final'] = {'exercise': exercise, 'confidence': round(final_conf, 1)}
        return exercise, round(final_conf, 1), debug

    def _stage1_posture_gate(self, f: Dict[str, float]) -> Dict[str, Any]:
        lying_score = f.get('posture_lying_score', 0.5)
        body_motion = f.get('body_vertical_motion', 0.5)
        torso_3d_mean = f.get('torso_pitch_3d_mean', 45.0)
        horiz_ratio = f.get('torso_horizontal_ratio', 0.5)

        reasons = []

        if torso_3d_mean > 60:
            reasons.append(f"LYING (3D): torso={torso_3d_mean:.1f}°(>60)")
            conf = 95.0 if body_motion < 0.15 else 88.0
            return {'is_lying': True, 'confidence': conf, 'reasons': reasons}

        if horiz_ratio > 2.0 and body_motion < 0.06:
            reasons.append(f"★ LYING (flat): horiz_ratio={horiz_ratio:.2f}(>2.0) + motion={body_motion:.3f}(<0.06)")
            return {'is_lying': True, 'confidence': 95.0, 'reasons': reasons}

        if horiz_ratio > 1.5 and body_motion < 0.04:
            reasons.append(f"LYING (flat-2): horiz_ratio={horiz_ratio:.2f}(>1.5) + motion={body_motion:.3f}(<0.04)")
            return {'is_lying': True, 'confidence': 90.0, 'reasons': reasons}

        if lying_score > 0.65 and body_motion < 0.12:
            reasons.append(f"LYING (2D): lying={lying_score:.2f}(>0.65) + motion={body_motion:.3f}(<0.12)")
            return {'is_lying': True, 'confidence': 88.0, 'reasons': reasons}

        if lying_score > 0.80:
            reasons.append(f"LYING (strong 2D): lying={lying_score:.2f}(>0.80)")
            return {'is_lying': True, 'confidence': 85.0, 'reasons': reasons}

        if body_motion > 0.15:
            reasons.append(f"STANDING: motion={body_motion:.3f}(>0.15)")
            return {'is_lying': False, 'confidence': 88.0, 'reasons': reasons}

        reasons.append("DEFAULT: standing")
        return {'is_lying': False, 'confidence': 75.0, 'reasons': reasons}

    def _stage2_motion_chain(self, f: Dict[str, float], stage1: Dict) -> Dict[str, Any]:
        upper_ratio = f.get('upper_motion_ratio', 0.5)
        lower_ratio = f.get('lower_motion_ratio', 0.5)
        knee_ratio = f.get('knee_motion_ratio', 0.3)
        hip_ratio = f.get('hip_motion_ratio', 0.3)
        is_lying = stage1.get('is_lying', False)

        elbow_rom = f.get('elbow_rom_bilateral', 0)
        body_motion = f.get('body_vertical_motion', 0.5)
        wrist_above_pct = f.get('wrist_above_shoulder_pct', 0.5)
        shoulder_energy = f.get('shoulder_angular_energy', 0)
        wrist_vert_dom = f.get('wrist_vertical_dominance', 0.5)

        reasons = []

        if elbow_rom > 40 and body_motion < 0.08 and wrist_above_pct < 0.2:
            reasons.append(f"★ BENCH FALLBACK: elbow_rom={elbow_rom:.0f}°(>40) + motion={body_motion:.3f}(<0.08) + wrist_above={wrist_above_pct:.2f}(<0.2)")
            return {'is_upper': True, 'motion_chain': 'bench', 'confidence': 95.0, 'bench_signal': True, 'reasons': reasons}

        if elbow_rom > 30 and body_motion < 0.10 and wrist_above_pct < 0.3:
            reasons.append(f"BENCH SOFT: elbow_rom={elbow_rom:.0f}°(>30) + motion={body_motion:.3f}(<0.10) + wrist_above={wrist_above_pct:.2f}(<0.3)")
            return {'is_upper': True, 'motion_chain': 'bench', 'confidence': 90.0, 'bench_signal': True, 'reasons': reasons}

        if is_lying:
            if (elbow_rom > 15 or shoulder_energy > 3 or wrist_vert_dom > 0.3 or upper_ratio > 0.35):
                reasons.append(f"LYING BENCH: lying=True, elbow_rom={elbow_rom:.1f}, shoulder_e={shoulder_energy:.2f}")
                return {'is_upper': True, 'motion_chain': 'bench', 'confidence': 95.0, 'bench_signal': True, 'reasons': reasons}
            reasons.append("LYING DEFAULT → UPPER")
            return {'is_upper': True, 'motion_chain': 'bench', 'confidence': 85.0, 'bench_signal': True, 'reasons': reasons}

        if upper_ratio > 0.55:
            reasons.append(f"UPPER: upper_ratio={upper_ratio:.3f}(>0.55)")
            return {'is_upper': True, 'motion_chain': 'upper', 'confidence': 85.0, 'bench_signal': False, 'reasons': reasons}

        if lower_ratio > 0.6:
            chain = 'knee_dominant' if knee_ratio > hip_ratio else 'hip_dominant'
            reasons.append(f"LOWER: lower={lower_ratio:.3f}(>0.6), knee={knee_ratio:.3f}, hip={hip_ratio:.3f} → {chain}")
            return {'is_upper': False, 'motion_chain': chain, 'confidence': 85.0, 'bench_signal': False, 'reasons': reasons}

        is_upper = upper_ratio > lower_ratio
        reasons.append(f"DEFAULT: upper={upper_ratio:.3f} vs lower={lower_ratio:.3f}")
        return {'is_upper': is_upper, 'motion_chain': 'upper' if is_upper else 'mixed', 'confidence': 65.0, 'bench_signal': False, 'reasons': reasons}

    def _stage3_upper(self, f: Dict[str, float], stage1: Dict, stage2: Dict) -> Dict[str, Any]:
        scores = {'Bench Press': 0.0, 'Overhead Press': 0.0}
        reasons = []

        is_lying = stage1.get('is_lying', False)
        bench_signal = stage2.get('bench_signal', False)
        body_motion = f.get('body_vertical_motion', 0.5)
        elbow_rom = f.get('elbow_rom_bilateral', 0)
        wrist_y_range = f.get('wrist_y_range_norm', 0)
        wrist_h_rom = f.get('wrist_horizontal_rom_norm', 0)
        wrist_above_pct = f.get('wrist_above_shoulder_pct', 0)

        if is_lying or bench_signal:
            scores['Bench Press'] += 35.0
            reasons.append(f"★ BENCH LOCK: lying={is_lying}, bench_signal={bench_signal} → +35")

        if body_motion < 0.03:
            scores['Bench Press'] += 10.0
            reasons.append(f"motion={body_motion:.3f}(<0.03) → Bench +10")
        elif body_motion < 0.08:
            scores['Bench Press'] += 5.0
            reasons.append(f"motion={body_motion:.3f}(<0.08) → Bench +5")
        elif body_motion > 0.15:
            scores['Overhead Press'] += 5.0
            reasons.append(f"motion={body_motion:.3f}(>0.15) → OHP +5")

        if elbow_rom > 40:
            scores['Bench Press'] += 15.0
            reasons.append(f"elbow_rom={elbow_rom:.0f}°(>40) → Bench +15")
        elif elbow_rom > 25:
            scores['Bench Press'] += 8.0
            reasons.append(f"elbow_rom={elbow_rom:.0f}°(>25) → Bench +8")

        if wrist_h_rom < 0.05:
            scores['Bench Press'] += 10.0
            reasons.append(f"wrist_h_rom={wrist_h_rom:.3f}(<0.05) → Bench +10")
        elif wrist_h_rom > 0.08:
            scores['Overhead Press'] += 5.0
            reasons.append(f"wrist_h_rom={wrist_h_rom:.3f}(>0.08) → OHP +5")

        if wrist_y_range > 0.4:
            scores['Overhead Press'] += 3.0
            reasons.append(f"wrist_y_range={wrist_y_range:.3f}(>0.4) → OHP +3")

        if wrist_above_pct > 0.7:
            scores['Overhead Press'] += 3.0
            reasons.append(f"wrist_above={wrist_above_pct:.2f}(>0.7) → OHP +3")
        if wrist_above_pct < 0.1:
            scores['Bench Press'] += 3.0
            reasons.append(f"wrist_above={wrist_above_pct:.2f}(<0.1) → Bench +3")

        best = max(scores, key=scores.get)
        total = sum(scores.values()) + 1e-8
        conf = min(max((scores[best] / total) * 100, 70.0), 99.0)

        return {'exercise': best, 'confidence': conf, 'scores': {k: round(v, 2) for k, v in scores.items()}, 'reasons': reasons}

    def _stage3_lower(self, f: Dict[str, float], stage1: Dict, stage2: Dict) -> Dict[str, Any]:
        lying_score = f.get('posture_lying_score', 0.0)
        is_lying = stage1.get('is_lying', False)
        body_motion = f.get('body_vertical_motion', 0.5)
        elbow_rom = f.get('elbow_rom_bilateral', 0)

        if body_motion < 0.03 and elbow_rom > 30:
            return {'exercise': 'Bench Press', 'confidence': 92.0, 'scores': {'Squat': 0, 'Deadlift': 0, 'Bench Press': 30}, 'reasons': [f"★ LOWER FUSE (motion+elbow): motion={body_motion:.3f}(<0.03) + elbow_rom={elbow_rom:.0f}°(>30) → 强制 Bench"]}

        if is_lying or lying_score > 0.7:
            return {'exercise': 'Bench Press', 'confidence': 90.0, 'scores': {'Squat': 0, 'Deadlift': 0, 'Bench Press': 30}, 'reasons': [f"LYING BLOCK: lying_score={lying_score:.2f}, is_lying={is_lying} → 强制 Bench"]}

        if elbow_rom > 50 and body_motion < 0.10:
            return {'exercise': 'Bench Press', 'confidence': 85.0, 'scores': {'Squat': 0, 'Deadlift': 0, 'Bench Press': 25}, 'reasons': [f"★ LOWER FUSE (strong elbow): elbow_rom={elbow_rom:.0f}°(>50) + motion={body_motion:.3f}(<0.10) → 强制 Bench"]}

        scores = {'Squat': 0.0, 'Deadlift': 0.0}
        reasons = []

        torso_mean = f.get('torso_pitch_3d_mean', 35.0)
        hip_hinge_mean = f.get('hip_hinge_mean', 45.0)
        hip_hinge_rom = f.get('hip_hinge_rom', 20.0)
        wrist_h_rom = f.get('wrist_horizontal_rom_norm', 0.0)
        knee_rom = f.get('knee_rom_bilateral', 0)
        hk_ratio = f.get('hip_knee_rom_ratio', 1.0)
        descent = f.get('descent_knee_leads_ratio', 0.5)
        shin_max = f.get('shin_angle_max', 20.0)

        if knee_rom > 50 and hk_ratio < 1.5 and torso_mean < 40:
            reasons.append(f"★ SQUAT PROTECTION: knee_rom={knee_rom:.0f}°(>50) + hk_ratio={hk_ratio:.2f}(<1.5) + torso_mean={torso_mean:.0f}°(<40)")
            return {'exercise': 'Squat', 'confidence': 92.0, 'scores': scores, 'reasons': reasons, 'squat_protection': True}

        if torso_mean < 35:
            scores['Squat'] += 8.0
            reasons.append(f"torso_mean={torso_mean:.1f}°(<35) → Squat +8")
        elif torso_mean > 45:
            scores['Deadlift'] += 10.0
            reasons.append(f"torso_mean={torso_mean:.1f}°(>45) → Deadlift +10")
        else:
            scores['Squat'] += 4.0 * (45 - torso_mean) / 10
            scores['Deadlift'] += 4.0 * (torso_mean - 35) / 10

        if hip_hinge_mean > 55:
            scores['Deadlift'] += 8.0
            reasons.append(f"hip_hinge_mean={hip_hinge_mean:.1f}°(>55) → Deadlift +8")
        elif hip_hinge_mean < 40:
            scores['Squat'] += 5.0
            reasons.append(f"hip_hinge_mean={hip_hinge_mean:.1f}°(<40) → Squat +5")
        else:
            scores['Squat'] += 2.5 * (55 - hip_hinge_mean) / 15
            scores['Deadlift'] += 2.5 * (hip_hinge_mean - 40) / 15

        if hip_hinge_rom > 35:
            scores['Deadlift'] += 5.0
            reasons.append(f"hip_hinge_rom={hip_hinge_rom:.1f}°(>35) → Deadlift +5")
        elif hip_hinge_rom < 20:
            scores['Squat'] += 3.0
            reasons.append(f"hip_hinge_rom={hip_hinge_rom:.1f}°(<20) → Squat +3")

        if wrist_h_rom > 0.12:
            scores['Deadlift'] += 5.0
            reasons.append(f"wrist_h_rom={wrist_h_rom:.3f}(>0.12) → Deadlift +5")
        elif wrist_h_rom < 0.05:
            scores['Squat'] += 3.0
            reasons.append(f"wrist_h_rom={wrist_h_rom:.3f}(<0.05) → Squat +3")

        if knee_rom > 60:
            scores['Squat'] += 5.0
            reasons.append(f"knee_rom={knee_rom:.0f}°(>60) → Squat +5")
        elif knee_rom < 40:
            scores['Deadlift'] += 4.0
            reasons.append(f"knee_rom={knee_rom:.0f}°(<40) → Deadlift +4")

        if descent > 0.6:
            scores['Squat'] += 3.0
            reasons.append(f"descent={descent:.2f}(>0.6) → Squat +3")
        elif descent < 0.4:
            scores['Deadlift'] += 3.0
            reasons.append(f"descent={descent:.2f}(<0.4) → Deadlift +3")

        if shin_max > 25:
            scores['Squat'] += 2.0
            reasons.append(f"shin_max={shin_max:.1f}°(>25) → Squat +2")
        elif shin_max < 15:
            scores['Deadlift'] += 2.0
            reasons.append(f"shin_max={shin_max:.1f}°(<15) → Deadlift +2")

        best = max(scores, key=scores.get)
        total = sum(scores.values()) + 1e-8
        conf = min(max((scores[best] / total) * 100, 55.0), 99.0)

        return {'exercise': best, 'confidence': conf, 'scores': {k: round(v, 2) for k, v in scores.items()}, 'reasons': reasons}