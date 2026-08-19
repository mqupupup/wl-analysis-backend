# backend/app/services/biomechanics/engine.py
"""
主引擎：整合所有模块，提供统一分析接口
V2 RepContext 架构 + V8 CycleRepDetector 兼容版
"""

import types
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from scipy.signal import savgol_filter

# ── V2 核心模块 ──
from app.domain.models import RepContext
from app.domain.enums import ValidationStatus, SignalSource
from app.services.biomechanics.kinematic_phase_engine import PhaseBuilder
from app.services.biomechanics.exercise_specific_scorer_v2 import (
    ExerciseSpecificScorerV2,
    RepScoreResult,
)
from app.services.biomechanics.error_detection_engine import (
    ErrorDetectionEngineFixed,
    SetLevelError,
)
from app.services.biomechanics.fatigue_analyzer import (
    FatigueAnalyzerFixed,
    FatigueResult,
)

from .pose_estimation import PoseEstimator, FramePoseData
from .feature_extraction import V24FeatureExtractor
from .movement_classification import V26HybridClassifier, PATTERN_TO_EXERCISE, MovementPattern
from .knowledge.loader import load_exercise
from .kinematic_phase_engine import CycleRepDetector


EXERCISE_ZH_MAP = {
    "Squat": "深蹲", "Bench Press": "卧推",
    "Deadlift": "硬拉", "Overhead Press": "过头推举",
    "Bicep Curl": "弯举",
    "Unknown": "未知动作"
}

SKELETON_CONNECTIONS = [
    (11, 12, "torso_top"), (23, 24, "torso_bottom"), (11, 23, "torso_left"), (12, 24, "torso_right"),
    (11, 13, "left_upper_arm"), (13, 15, "left_forearm"),
    (12, 14, "right_upper_arm"), (14, 16, "right_forearm"),
    (23, 25, "left_thigh"), (25, 27, "left_shin"),
    (24, 26, "right_thigh"), (26, 28, "right_shin"),
]


class BiomechanicsEngine:
    """生物力学分析引擎"""

    def __init__(self):
        self.pose_estimator = PoseEstimator()
        self.classifier = V26HybridClassifier()
        self._last_debug = {}
        self._last_features = {}

    def analyze(self, video_path: Path) -> Dict[str, Any]:
        """主分析入口"""
        print(f"\n{'='*70}")
        print(f"🎬 开始分析: {video_path.name}")
        print(f"{'='*70}")

        # 1. 姿态估计
        frame_data_list = self.pose_estimator.process_video(video_path)
        print(f"📹 姿态帧数: {len(frame_data_list)}")
        if len(frame_data_list) < 10:
            return {"success": False, "error": "未检测到足够的有效姿态数据"}

        fps = self.pose_estimator.fps
        frame_height = self.pose_estimator.frame_height
        print(f"⚙️ FPS: {fps}, 帧高度: {frame_height}")

        # 2. 特征提取
        feature_extractor = V24FeatureExtractor(fps=fps, frame_height=frame_height)
        features = feature_extractor.extract(frame_data_list)
        self._last_features = features
        print(f"🔢 提取特征数: {len(features)}")

        # 3. 动作分类
        pattern, conf = self._classify_movement(features)
        exercise = PATTERN_TO_EXERCISE.get(pattern, "Unknown")
        print(f"🏷️ 识别动作: {exercise} (置信度: {conf:.1f}%)")
        if exercise == "Unknown":
            return {"success": False, "error": "未检测到标准动作"}

        # 4. 加载知识库
        knowledge = load_exercise(exercise)
        if not knowledge:
            print(f"⚠️ 未找到 {exercise} 的知识库，使用通用分析")
            knowledge = self._build_fallback_knowledge(exercise)
        else:
            print(f"📚 知识库加载成功: {len(knowledge.get('errors', []))} 条错误规则")

        # ══════════════════════════════════════════════════════
        # 5. V8 主角度提取 + V7 Cycle Rep 检测
        # ══════════════════════════════════════════════════════
        primary_angle_key = knowledge.get("rep_detection", {}).get(
            "primary_angle", "elbow_angle_avg"
        )
        raw_angles = self._resolve_primary_angles(frame_data_list, primary_angle_key, fps)

        rep_config = knowledge.get("rep_detection", {})

        # ── V7 修改点 1: 覆盖知识库中的旧参数 ──
        # 知识库 JSON 中可能残留 V3 时代的 min_distance_sec=0.8 / min_prominence=10.0
        # 这里显式注入 V7 推荐值，确保不被旧配置覆盖
        rep_config = self._apply_v7_defaults(rep_config)

        cycle_detector = CycleRepDetector(fps=fps, config=rep_config)

        # ── V7 修改点 2: 删除冗余的 find_peaks 预统计 ──
        # 旧代码用 cycle_detector.min_distance_sec / min_prominence 重新跑 find_peaks，
        # 得到的 "Cycle candidates: 27" 与 V7 内部的候选数不一致，造成混淆。
        # V7 的 detect() 内部已打印完整的候选生成 + 评分日志，无需额外预统计。

        detected_reps = cycle_detector.detect(raw_angles)

        print(f"\n📊 V8 CycleRepDetector (v7) 结果:")
        print(f"   validated reps: {len(detected_reps)}")

        # ══════════════════════════════════════════════════════
        # 6. V2 PhaseBuilder → RepContext[]
        # ══════════════════════════════════════════════════════
        min_rom = knowledge.get("rep_detection", {}).get("min_rom_no_pause", 40.0)
        reject_rom = knowledge.get("rep_detection", {}).get("min_rom_existence", 15.0)

        phase_builder = PhaseBuilder(
            fps=fps,
            min_rom_deg=min_rom,
            reject_rom_deg=reject_rom,
        )

        # 提取单侧肘角序列供 RepContext 使用
        left_elbow_arr, right_elbow_arr = self._extract_side_elbow_arrays(frame_data_list)
        # 提取 wrist 坐标供 bar path 检测使用
        left_wrist_arr, right_wrist_arr = self._extract_side_wrist_arrays(frame_data_list)
        # 提取 upper_arm_torso 角度供 elbow tuck 检测使用
        left_uat_arr, right_uat_arr = self._extract_side_upper_arm_torso_arrays(frame_data_list)

        # ── V7 修改点 3: 修复 both_valid 未定义的 Bug ──
        n_frames = len(frame_data_list)
        both_valid = 0
        if left_elbow_arr is not None and right_elbow_arr is not None:
            both_valid = int(np.sum(np.isfinite(left_elbow_arr) & np.isfinite(right_elbow_arr)))
            bilateral_valid_ratio = both_valid / max(n_frames, 1)
        else:
            bilateral_valid_ratio = 0.0

        bilateral_min_ratio = rep_config.get("bilateral_min_ratio", 0.65)
        left_valid = int(np.sum(np.isfinite(left_elbow_arr))) if left_elbow_arr is not None else 0
        right_valid = int(np.sum(np.isfinite(right_elbow_arr))) if right_elbow_arr is not None else 0

        if bilateral_valid_ratio >= bilateral_min_ratio:
            signal_source = "bilateral"
        elif left_valid >= right_valid:
            signal_source = "left"
        else:
            signal_source = "right"

        print(f"📊 双侧信号质量: 双侧={both_valid} 左={left_valid} 右={right_valid} | "
              f"bilateral_ratio={bilateral_valid_ratio:.3f} 阈值={bilateral_min_ratio} → signal_source={signal_source}")

        contexts: List[RepContext] = []
        for i, det in enumerate(detected_reps):
            # ── V7 修改点 4: 修复 start_frame=0 时 `or` 误判的 Bug ──
            sf = self._get_frame(det, "start_frame", 0)
            bf = self._get_frame(det, "bottom_frame", 0)
            ef = self._get_frame(det, "end_frame", 0)

            ctx = phase_builder.build(
                rep_index=i + 1,
                start_frame=sf,
                bottom_frame=bf,
                end_frame=ef,
                angles=raw_angles,
                left_elbow=left_elbow_arr,
                right_elbow=right_elbow_arr,
                bilateral_valid_ratio=bilateral_valid_ratio,
                signal_source=signal_source,
                left_wrist=left_wrist_arr,
                right_wrist=right_wrist_arr,
                left_upper_arm_torso=left_uat_arr,
                right_upper_arm_torso=right_uat_arr,
            )
            contexts.append(ctx)

        valid_contexts = [c for c in contexts if c.validation_status == ValidationStatus.VALID]

        print(f"\n📊 V2 PhaseBuilder 结果:")
        for ctx in contexts:
            phases_info = ", ".join([p.name for p in ctx.phases])
            print(f"   - Rep {ctx.rep_index}: [{ctx.start_frame}-{ctx.end_frame}] "
                  f"ROM={ctx.actual_rom:.1f}° status={ctx.validation_status.value} "
                  f"phases=[{phases_info}]")
        print(f"   ✅ 有效 Rep: {len(valid_contexts)}/{len(contexts)}")

        # ══════════════════════════════════════════════════════
        # 7. V2 Error Detection
        # ══════════════════════════════════════════════════════
        error_engine = ErrorDetectionEngineFixed()
        set_errors = error_engine.detect_set(contexts)

        print(f"\n⚠️ V2 错误检测结果:")
        print(f"   - 检测到 {len(set_errors)} 类动作错误")
        for err in set_errors:
            print(f"   - {err.display_name}: {len(err.occurrences)}/{len(valid_contexts)} "
                  f"(severity={err.worst_severity.value if err.worst_severity else 'n/a'})")

        # ══════════════════════════════════════════════════════
        # 8. V2 Scoring
        # ══════════════════════════════════════════════════════
        scorer = ExerciseSpecificScorerV2()
        rep_scores: List[RepScoreResult] = [scorer.score_rep(c) for c in valid_contexts]

        print(f"\n📈 V2 质量评分结果:")
        for s in rep_scores:
            print(f"   - Rep {s.rep_index}: overall={s.overall_score} "
                  f"technique={s.technique_score} status={s.status.value}")

        # ══════════════════════════════════════════════════════
        # 9. V2 Fatigue Analysis
        # ══════════════════════════════════════════════════════
        fatigue_analyzer = FatigueAnalyzerFixed()
        fatigue_result = fatigue_analyzer.analyze(contexts)

        print(f"\n💪 V2 疲劳分析:")
        print(f"   - 速度损失: {fatigue_result.velocity_loss_pct:.1f}%")
        print(f"   - 疲劳等级: {fatigue_result.fatigue_level}")
        print(f"   - 预估 RIR: {fatigue_result.estimated_rir}")

        # ══════════════════════════════════════════════════════
        # 10. 汇总输出（兼容原有前端格式）
        # ══════════════════════════════════════════════════════
        avg_score = round(float(np.mean([s.overall_score for s in rep_scores
                                         if s.overall_score is not None])), 1) if rep_scores else 0.0

        avg_data_quality = round(float(np.mean([
            getattr(s, "data_quality_score", 100.0) for s in rep_scores
        ])), 1) if rep_scores else 100.0

        error_summary = self._summarize_v2_errors(set_errors, valid_contexts)
        feedback = self._generate_v2_feedback(rep_scores, set_errors, exercise)
        skeleton_frames = self._build_skeleton_frames(frame_data_list, target_count=8)
        angle_curves = self._extract_angle_curves(frame_data_list)

        # 兼容旧版 v2_reps 格式
        v2_formatted_reps = self._format_v2_reps_for_frontend(rep_scores, contexts)

        print(f"\n{'='*70}")
        print(f"✅ 分析完成: {exercise}, {len(contexts)} 次, "
              f"有效 {len(valid_contexts)}, 评分 {avg_score}, 错误 {len(set_errors)} 类")
        print(f"{'='*70}\n")

        return {
            "success": True,
            "exercise_type": exercise,
            "exercise_type_zh": EXERCISE_ZH_MAP.get(exercise, "未知"),
            "confidence": round(conf, 1),
            "rep_count": len(contexts),
            "reps_detail": [
                {
                    "rep_index": ctx.rep_index,
                    "duration": round(ctx.total_duration, 2),
                    "actual_rom": round(ctx.actual_rom, 1),
                    "validation_status": ctx.validation_status.value,
                    "phases": [
                        {
                            "name": p.name,
                            "duration": round(p.end_time - p.start_time, 2),
                            "start_frame": p.start_frame,
                            "end_frame": p.end_frame,
                        } for p in ctx.phases
                    ],
                    "quality_score": next(
                        (s.overall_score for s in rep_scores if s.rep_index == ctx.rep_index), None
                    ),
                    "data_quality": next(
                        (getattr(s, "data_quality_score", 100.0) for s in rep_scores if s.rep_index == ctx.rep_index), 100.0
                    ),
                    "feedback": "",
                } for ctx in contexts
            ],
            "quality": {
                "total_score": avg_score,
                "data_quality": avg_data_quality,
                "breakdown": self._summarize_v2_layers(rep_scores),
                "feedback": feedback,
            },
            "errors": error_summary,
            "fatigue": {
                "velocity_loss_pct": fatigue_result.velocity_loss_pct,
                "estimated_rir": fatigue_result.estimated_rir,
                "fatigue_level": fatigue_result.fatigue_level,
                "trend": fatigue_result.trend,
                "status": fatigue_result.status,
                "velocity_curve": [],
            },
            "skeleton_frames": skeleton_frames,
            "angle_curves": angle_curves,
            "summary_metrics": {
                "duration_sec": round(len(frame_data_list) / fps, 1),
                "fps_detected": round(fps, 1),
            },
            "v2_reps": v2_formatted_reps,
        }

    # =========================================================================
    # V7 新增辅助方法
    # =========================================================================

    @staticmethod
    def _get_frame(det, key: str, default: int = 0) -> int:
        """
        安全获取帧号，修复 `or` 对 0 值的误判。

        旧代码: getattr(det, "start_frame", None) or det.get("start_frame", 0)
        问题:   当 start_frame=0 时，`or` 会跳过 attribute 走 dict 分支。
        """
        if isinstance(det, dict):
            return int(det.get(key, default))
        val = getattr(det, key, None)
        return int(val) if val is not None else default

    @staticmethod
    def _apply_v7_defaults(rep_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        为知识库配置注入 V7 推荐默认值。

        知识库 JSON 中可能残留 V3 时代的旧参数（如 min_distance_sec=0.8），
        这里用 V7 推荐值覆盖，但保留知识库中明确设置的 V7 新参数。
        """
        cfg = dict(rep_config)  # 浅拷贝，不修改原始知识库

        # V7.2 核心参数覆盖（旧值 → 新值）
        v7_overrides = {
            "min_distance_sec": 0.6,        # V3: 0.8 → V7: 0.6
            "min_prominence": 10.0,         # V7: 8.0 → V7.2: 10.0（13过度收紧）
            "min_eccentric_duration_s": 0.35,  # V7: 0.20 → V7.1: 0.35
            "min_concentric_duration_s": 0.30, # V7: 0.20 → V7.1: 0.30
            "min_bottom_to_bottom_frames": 30, # V7: 20 → V7.1: 30
            "max_closure_deg": 55.0,        # V7: 60.0 → V7.1: 55.0
            "ideal_min_rom_deg": 55.0,
            "min_chain_score": 0.35,        # V7: 0.25 → V7.1: 0.35
            # V7.2 新增参数
            "valid_min_rom_deg": 48.0,
            "max_rom_deg": 100.0,
            "min_start_gap_sec": 0.4,
            "rerack_long_next_gap_s": 3.0,
            "min_consecutive_reps": 3,
            "max_set_gap_sec": 4.0,
            "pre_rest_threshold_s": 2.0,
            "tail_ghost_enabled": True,
            "tail_ghost_max_gap_sec": 1.2,
            "tail_ghost_min_strong_score": 0.75,
            "tail_ghost_max_weak_score": 0.65,
            "tail_ghost_rom_ratio": 0.72,
            "tail_ghost_prom_ratio": 0.55,
            "bridge_enabled": True,
            "bridge_boundary_frames": 2,
            "bridge_max_score": 0.60,
            "bridge_max_prominence": 15.0,
            "bridge_min_score_gap": 0.20,
            "bridge_min_strong_score": 0.75,
            "setup_cycle_enabled": True,
            "setup_cycle_edge_frames": 90,
            "setup_cycle_min_gap_sec": 2.5,
            "setup_cycle_max_closure_deg": 55.0,
            "setup_cycle_min_next_score": 0.55,
            "setup_cycle_min_next_prominence": 12.0,
        }

        for key, val in v7_overrides.items():
            cfg[key] = val  # 强制覆盖旧值

        return cfg

    # =========================================================================
    # V2 辅助方法
    # =========================================================================

    def _extract_side_elbow_arrays(
        self, frame_data_list: List[FramePoseData]
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """从 FramePoseData 提取左右肘角数组"""
        n = len(frame_data_list)
        left = np.full(n, np.nan, dtype=np.float64)
        right = np.full(n, np.nan, dtype=np.float64)

        for i, fd in enumerate(frame_data_list):
            angles = getattr(fd, "angles", {}) or {}
            lv = angles.get("left_elbow")
            rv = angles.get("right_elbow")
            if lv is not None and np.isfinite(lv):
                left[i] = float(lv)
            if rv is not None and np.isfinite(rv):
                right[i] = float(rv)

        l_valid = int(np.sum(np.isfinite(left)))
        r_valid = int(np.sum(np.isfinite(right)))

        left_out = left if l_valid > 0 else None
        right_out = right if r_valid > 0 else None
        return left_out, right_out

    def _extract_side_wrist_arrays(
        self, frame_data_list: List[FramePoseData]
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """从 FramePoseData 提取左右 wrist 坐标数组（形状 N×2，用于 bar path）"""
        n = len(frame_data_list)
        left = np.full((n, 2), np.nan, dtype=np.float64)
        right = np.full((n, 2), np.nan, dtype=np.float64)

        WRIST_ALIASES = {
            "left": ["left_wrist", "LEFT_WRIST", "15"],
            "right": ["right_wrist", "RIGHT_WRIST", "16"],
        }

        def _get_pos(fd, side):
            positions = getattr(fd, "positions", {}) or {}
            for alias in WRIST_ALIASES[side]:
                v = positions.get(alias)
                if v is not None and len(v) >= 2:
                    return float(v[0]), float(v[1])
            return None, None

        for i, fd in enumerate(frame_data_list):
            lx, ly = _get_pos(fd, "left")
            rx, ry = _get_pos(fd, "right")
            if lx is not None:
                left[i, 0] = lx
                left[i, 1] = ly
            if rx is not None:
                right[i, 0] = rx
                right[i, 1] = ry

        l_valid = int(np.sum(np.isfinite(left[:, 0])))
        r_valid = int(np.sum(np.isfinite(right[:, 0])))

        left_out = left if l_valid > 0 else None
        right_out = right if r_valid > 0 else None
        return left_out, right_out

    def _extract_side_upper_arm_torso_arrays(
        self, frame_data_list: List[FramePoseData]
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """从 FramePoseData 提取左右 upper_arm_torso 角度序列（由 feature_extractor 计算并存入 fd.angles）"""
        n = len(frame_data_list)
        left = np.full(n, np.nan, dtype=np.float64)
        right = np.full(n, np.nan, dtype=np.float64)

        for i, fd in enumerate(frame_data_list):
            angles = getattr(fd, "angles", {}) or {}
            lv = angles.get("left_upper_arm_torso")
            rv = angles.get("right_upper_arm_torso")
            if lv is not None and np.isfinite(lv):
                left[i] = float(lv)
            if rv is not None and np.isfinite(rv):
                right[i] = float(rv)

        l_valid = int(np.sum(np.isfinite(left)))
        r_valid = int(np.sum(np.isfinite(right)))

        left_out = left if l_valid > 0 else None
        right_out = right if r_valid > 0 else None
        return left_out, right_out

    def _summarize_v2_errors(
        self, set_errors: List[SetLevelError], valid_contexts: List[RepContext]
    ) -> Dict:
        """将 V2 SetLevelError 转为前端兼容格式"""
        if not set_errors:
            return {"total_count": 0, "by_severity": {}, "top_errors": []}

        by_severity: Dict[str, int] = {}
        for e in set_errors:
            sev = e.worst_severity.value if e.worst_severity else "moderate"
            by_severity[sev] = by_severity.get(sev, 0) + 1

        top_errors = []
        for e in sorted(set_errors, key=lambda x: len(x.occurrences), reverse=True)[:3]:
            top_errors.append({
                "error_id": e.error_id,
                "name_zh": e.display_name,
                "severity": e.worst_severity.value if e.worst_severity else "moderate",
                "feedback": f"{e.display_name} 出现在 {len(e.occurrences)}/{len(valid_contexts)} 次动作中",
            })

        return {
            "total_count": len(set_errors),
            "by_severity": by_severity,
            "top_errors": top_errors,
        }

    def _summarize_v2_layers(self, rep_scores: List[RepScoreResult]) -> Dict:
        """汇总各层评分"""
        if not rep_scores:
            return {}

        layer_names = ["technique", "movement_quality", "safety", "performance"]
        result = {}
        for ln in layer_names:
            vals = [getattr(s, f"{ln}_score", None) or s.layers.get(ln).score
                    for s in rep_scores
                    if (getattr(s, f"{ln}_score", None) is not None
                        or (ln in s.layers and s.layers[ln].score is not None))]
            if vals:
                result[ln] = {
                    "score": round(float(np.mean(vals)), 1),
                    "name_zh": {"technique": "技术", "movement_quality": "动作质量",
                                "safety": "安全性", "performance": "表现力"}.get(ln, ln),
                    "name_en": ln,
                }
        return result

    def _generate_v2_feedback(
        self, rep_scores: List[RepScoreResult], set_errors: List[SetLevelError], exercise: str
    ) -> List[str]:
        """生成总体反馈"""
        feedback = []
        if not rep_scores:
            feedback.append("📋 未检测到完整的动作周期")
            return feedback

        avg = float(np.mean([s.overall_score for s in rep_scores if s.overall_score is not None]))
        zh = EXERCISE_ZH_MAP.get(exercise, exercise)
        cnt = len(rep_scores)

        if avg >= 9.0:
            feedback.append(f"🏆 {zh}技术非常出色！{cnt}次动作质量稳定。")
        elif avg >= 7.5:
            feedback.append(f"👍 {zh}整体不错，有改进空间。")
        elif avg >= 6.0:
            feedback.append(f"🔧 {zh}需要针对性改进。")
        else:
            feedback.append(f"⛔ {zh}存在安全隐患，建议降低重量。")

        severe_errors = [e for e in set_errors if e.worst_severity and e.worst_severity.value == "severe"]
        if severe_errors:
            feedback.append(f"🚨 {severe_errors[0].display_name} 频繁出现，请重点关注。")

        return feedback

    def _format_v2_reps_for_frontend(
        self, rep_scores: List[RepScoreResult], contexts: List[RepContext]
    ) -> List[Dict]:
        """兼容旧版 v2_reps 输出格式"""
        results = []
        for s in rep_scores:
            ctx = next((c for c in contexts if c.rep_index == s.rep_index), None)
            rep_dict = {
                "rep_index": s.rep_index,
                "overall_score": s.overall_score,
                "technique_score": s.technique_score,
                "status": s.status.value,
                "data_quality": getattr(s, "data_quality_score", 100.0),
                "grade": getattr(s, "grade", "N/A"),
                "layers": {},
            }
            for ln, lr in s.layers.items():
                rep_dict["layers"][ln] = {
                    "score": lr.score,
                    "status": lr.status.value,
                    "metrics": [
                        {"key": m.key, "raw": m.raw, "score": m.score,
                         "status": m.status.value, "detail": getattr(m, "detail", "")}
                        for m in lr.metrics
                    ],
                }
            if ctx:
                rep_dict["actual_rom"] = round(ctx.actual_rom, 1)
                rep_dict["duration"] = round(ctx.total_duration, 2)
                rep_dict["validation_status"] = ctx.validation_status.value
                rep_dict["bilateral_valid_ratio"] = round(ctx.bilateral_valid_ratio, 3)
                rep_dict["signal_source"] = ctx.signal_source.value if hasattr(ctx.signal_source, "value") else str(ctx.signal_source)
            results.append(rep_dict)
        return results

    # =========================================================================
    # 原有方法（完全保留）
    # =========================================================================

    def _resolve_primary_angles(
        self,
        frame_data_list: List[FramePoseData],
        primary_angle_key: str,
        fps: float
    ) -> np.ndarray:
        """双侧融合 → NaN插值 → 异常值剔除 → SG平滑"""
        n = len(frame_data_list)
        if n == 0:
            return np.array([], dtype=np.float64)

        SIDE_MAP = {
            "elbow_angle_avg": ("left_elbow", "right_elbow"),
            "shoulder_flexion_avg": ("left_shoulder", "right_shoulder"),
            "knee_angle_avg": ("left_knee", "right_knee"),
            "hip_angle_avg": ("left_hip", "right_hip"),
        }

        if primary_angle_key in SIDE_MAP:
            left_key, right_key = SIDE_MAP[primary_angle_key]
            values = np.full(n, np.nan, dtype=np.float64)

            both = left_only = right_only = missing = 0
            for i, fd in enumerate(frame_data_list):
                angles = getattr(fd, "angles", {}) or {}
                l_val = angles.get(left_key)
                r_val = angles.get(right_key)

                l_ok = l_val is not None and np.isfinite(l_val) and 20.0 <= float(l_val) <= 180.0
                r_ok = r_val is not None and np.isfinite(r_val) and 20.0 <= float(r_val) <= 180.0

                if l_ok and r_ok:
                    values[i] = (float(l_val) + float(r_val)) / 2.0
                    both += 1
                elif l_ok:
                    values[i] = float(l_val)
                    left_only += 1
                elif r_ok:
                    values[i] = float(r_val)
                    right_only += 1
                else:
                    missing += 1

            valid_count = int(np.sum(np.isfinite(values)))
            print(f"📐 主角度融合: {primary_angle_key} | 双侧={both} 左={left_only} 右={right_only} 缺={missing} | 有效={valid_count}/{n}")

            if valid_count < max(10, int(n * 0.3)):
                print(f"⚠️ {primary_angle_key} 有效数据不足，尝试从 positions 实时合成")
                return self._resolve_elbow_from_positions(frame_data_list, fps)
        else:
            values = np.array([
                float(fd.angles.get(primary_angle_key))
                if fd.angles.get(primary_angle_key) is not None and np.isfinite(fd.angles.get(primary_angle_key))
                else np.nan
                for fd in frame_data_list
            ], dtype=np.float64)

        values = self._interpolate_angle_series(values, fps)
        values = self._robust_clip_angle_series(values)
        values = self._smooth_angles(values, fps)

        return values

    def _resolve_elbow_from_positions(self, frame_data_list: List[FramePoseData], fps: float) -> np.ndarray:
        """从 positions 实时合成肘角（兜底）"""
        n = len(frame_data_list)
        KEY_ALIASES = {
            "left_shoulder": ["left_shoulder", "LEFT_SHOULDER", "11"],
            "right_shoulder": ["right_shoulder", "RIGHT_SHOULDER", "12"],
            "left_elbow": ["left_elbow", "LEFT_ELBOW", "13"],
            "right_elbow": ["right_elbow", "RIGHT_ELBOW", "14"],
            "left_wrist": ["left_wrist", "LEFT_WRIST", "15"],
            "right_wrist": ["right_wrist", "RIGHT_WRIST", "16"],
        }

        def _get_pos(fd, key):
            for alias in KEY_ALIASES.get(key, [key]):
                v = fd.positions.get(alias)
                if v is not None:
                    return v
            return None

        def _angle_2d(a, b, c):
            if a is None or b is None or c is None:
                return None
            ba = np.array([a[0] - b[0], a[1] - b[1]], dtype=float)
            bc = np.array([c[0] - b[0], c[1] - b[1]], dtype=float)
            nb, nc = np.linalg.norm(ba), np.linalg.norm(bc)
            if nb < 1e-8 or nc < 1e-8:
                return None
            cos_v = np.clip(np.dot(ba, bc) / (nb * nc), -1.0, 1.0)
            return float(np.degrees(np.arccos(cos_v)))

        angle_defs = [
            ("left_shoulder", "left_elbow", "left_wrist"),
            ("right_shoulder", "right_elbow", "right_wrist"),
        ]

        synthesized = []
        for fd in frame_data_list:
            vals = []
            for pa, pb, pc in angle_defs:
                a, b, c = _get_pos(fd, pa), _get_pos(fd, pb), _get_pos(fd, pc)
                v = _angle_2d(a, b, c)
                if v is not None:
                    vals.append(v)
            if vals:
                synthesized.append(float(np.mean(vals)))
            else:
                synthesized.append(np.nan)

        values = np.array(synthesized, dtype=np.float64)
        valid_count = int(np.sum(np.isfinite(values)))
        print(f"📐 Positions 合成肘角: 有效={valid_count}/{n}")

        values = self._interpolate_angle_series(values, fps)
        values = self._robust_clip_angle_series(values)
        values = self._smooth_angles(values, fps)
        return values

    def _interpolate_angle_series(self, values: np.ndarray, fps: float = 30.0) -> np.ndarray:
        """
        Linear interpolation with max gap limit.
        Short NaN (<=0.2s): linear interp; Long NaN (>0.2s): mean fill (constant, no fake extrema)
        """
        values = values.astype(np.float64).copy()
        valid = np.isfinite(values)
        if valid.sum() < 2:
            return values

        n = len(values)
        indices = np.arange(n)
        max_gap_frames = max(1, int(round(0.20 * fps)))

        values[~valid] = np.interp(indices[~valid], indices[valid], values[valid])

        nan_mask = ~valid
        if nan_mask.any():
            in_gap = False
            gap_start = 0
            for i in range(n + 1):
                is_nan = i < n and nan_mask[i]
                if is_nan and not in_gap:
                    gap_start = i
                    in_gap = True
                elif not is_nan and in_gap:
                    gap_end = i
                    if (gap_end - gap_start) > max_gap_frames:
                        lv = values[gap_start - 1] if gap_start > 0 else np.nan
                        rv = values[gap_end] if gap_end < n else np.nan
                        if np.isfinite(lv) and np.isfinite(rv):
                            fv = (lv + rv) / 2.0
                        elif np.isfinite(lv):
                            fv = lv
                        elif np.isfinite(rv):
                            fv = rv
                        else:
                            fv = np.nanmean(values)
                        values[gap_start:gap_end] = fv
                    in_gap = False

        return values

    def _robust_clip_angle_series(self, values: np.ndarray) -> np.ndarray:
        """基于 MAD 的 MediaPipe 突刺剔除"""
        values = values.astype(np.float64).copy()
        if len(values) < 10:
            return values

        diff = np.diff(values)
        median_diff = np.median(diff)
        mad = np.median(np.abs(diff - median_diff))
        threshold = max(12.0, 6.0 * mad)

        bad = np.zeros(len(values), dtype=bool)
        for i in range(1, len(values)):
            if abs(diff[i - 1] - median_diff) > threshold:
                bad[i] = True

        if bad.any():
            good = ~bad
            if good.sum() >= 2:
                indices = np.arange(len(values))
                values[bad] = np.interp(indices[bad], indices[good], values[good])
        return values

    def _smooth_angles(self, angles: np.ndarray, fps: float) -> np.ndarray:
        """角度平滑 — 只做一次轻度 SG（~0.12s）"""
        if len(angles) < 9:
            return angles.astype(np.float64)

        result = angles.astype(np.float64).copy()

        try:
            window = max(5, int(round(fps * 0.12)))
            if window % 2 == 0:
                window += 1
            max_window = len(result) if len(result) % 2 == 1 else len(result) - 1
            window = min(window, max_window)

            if window >= 5:
                result = savgol_filter(
                    result, window_length=window, polyorder=2
                )
        except Exception:
            pass

        return result.astype(np.float64)

    def _classify_movement(self, features):
        """分类动作模式"""
        exercise, confidence, debug = self.classifier.classify(features)
        self._last_debug = debug

        print(f"\n{'=' * 70}")
        print(f"🔬 V26 分类器")
        print(f"{'=' * 70}")

        if 'stage1' in debug:
            s1 = debug['stage1']
            print(f"\n📌 Stage 1: is_lying={s1.get('is_lying')} ({s1.get('confidence', 0):.1f}%)")
            for r in s1.get('reasons', []):
                print(f"   ✓ {r}")

        if 'stage2' in debug:
            s2 = debug['stage2']
            print(f"\n📌 Stage 2: is_upper={s2.get('is_upper')}, chain={s2.get('motion_chain')}, bench_signal={s2.get('bench_signal')}")
            for r in s2.get('reasons', []):
                print(f"   ✓ {r}")

        if 'stage3' in debug:
            s3 = debug['stage3']
            print(f"\n📌 Stage 3: {s3.get('exercise')} ({s3.get('confidence', 0):.1f}%)")
            if 'scores' in s3:
                print(f"   评分: {s3['scores']}")
            for r in s3.get('reasons', []):
                print(f"   ✓ {r}")

        print(f"\n✅ 最终: {exercise} ({confidence:.1f}%)")
        print(f"{'=' * 70}\n")

        exercise_to_pattern = {
            'Squat': MovementPattern.LOWER_BODY_SQUAT,
            'Deadlift': MovementPattern.LOWER_BODY_HINGE,
            'Bench Press': MovementPattern.UPPER_BODY_HORIZONTAL_PUSH,
            'Overhead Press': MovementPattern.UPPER_BODY_VERTICAL_PUSH,
        }
        return exercise_to_pattern.get(exercise, MovementPattern.UNKNOWN), confidence

    def _build_fallback_knowledge(self, exercise: str) -> Dict:
        """构建兜底知识库 — V7 参数"""
        primary_angle_map = {
            "Bench Press": "elbow_angle_avg",
            "Overhead Press": "shoulder_flexion_avg",
            "Squat": "knee_angle_avg",
            "Deadlift": "hip_angle",
            "Bicep Curl": "elbow_angle_avg",
        }
        return {
            "meta": {"id": exercise.lower(), "name_zh": EXERCISE_ZH_MAP.get(exercise, exercise)},
            "phases": {
                "states": ["setup", "eccentric", "concentric", "lockout"],
                "transitions": []
            },
            "key_points": [],
            "errors": [],
            "rep_detection": {
                "primary_angle": primary_angle_map.get(exercise, "knee_angle_avg"),
                # ── V7.2 参数 ──
                "min_distance_sec": 0.6,
                "min_prominence": 10.0,
                "min_eccentric_duration_s": 0.35,
                "min_concentric_duration_s": 0.30,
                "min_bottom_to_bottom_frames": 30,
                "max_closure_deg": 55.0,
                "ideal_min_rom_deg": 55.0,
                "min_chain_score": 0.35,
                "valid_min_rom_deg": 48.0,
                "max_rom_deg": 100.0,
                "min_start_gap_sec": 0.4,
                "rerack_long_next_gap_s": 3.0,
                "min_consecutive_reps": 3,
                "max_set_gap_sec": 4.0,
                "pre_rest_threshold_s": 2.0,
                "tail_ghost_enabled": True,
                "tail_ghost_max_gap_sec": 1.2,
                "tail_ghost_min_strong_score": 0.75,
                "tail_ghost_max_weak_score": 0.65,
                "tail_ghost_rom_ratio": 0.72,
                "tail_ghost_prom_ratio": 0.55,
                "bridge_enabled": True,
                "bridge_boundary_frames": 2,
                "bridge_max_score": 0.60,
                "bridge_max_prominence": 15.0,
                "bridge_min_score_gap": 0.20,
                "bridge_min_strong_score": 0.75,
                "setup_cycle_enabled": True,
                "setup_cycle_edge_frames": 90,
                "setup_cycle_min_gap_sec": 2.5,
                "setup_cycle_max_closure_deg": 55.0,
                "setup_cycle_min_next_score": 0.55,
                "setup_cycle_min_next_prominence": 12.0,
                # ── 保留原有参数 ──
                "min_rom_existence": 15.0,
                "min_duration_existence": 0.35,
                "min_direction_consistency": 0.30,
                "min_rom_no_pause": 40.0,
                "quality_min_top_recovery": 82.0,
                "max_duration": 8.0,
                "debug_frame_start": 400,
                "debug_frame_end": 850,
            },
            "feedback_rules": {}
        }

    def _build_skeleton_frames(self, frame_data_list, target_count=8):
        """构建骨架帧可视化数据"""
        if not frame_data_list:
            return []
        step = max(1, len(frame_data_list) // target_count)
        key_frames = []
        mapping = {11: 'left_shoulder', 12: 'right_shoulder', 13: 'left_elbow', 14: 'right_elbow',
                   15: 'left_wrist', 16: 'right_wrist', 23: 'left_hip', 24: 'right_hip',
                   25: 'left_knee', 26: 'right_knee', 27: 'left_ankle', 28: 'right_ankle'}
        for i in range(0, len(frame_data_list), step):
            if len(key_frames) >= target_count:
                break
            fd = frame_data_list[i]
            bones = []
            for si, ei, bn in SKELETON_CONNECTIONS:
                sn, en = mapping.get(si), mapping.get(ei)
                if sn in fd.positions and en in fd.positions:
                    bones.append({"name": bn,
                                  "start": [round(fd.positions[sn][0]), round(fd.positions[sn][1])],
                                  "end": [round(fd.positions[en][0]), round(fd.positions[en][1])]})
            joints = [{"name": n, "position": [round(p[0]), round(p[1])]}
                      for n, p in fd.positions.items()]
            key_frames.append({
                "frame_idx": fd.frame_idx,
                "pct": round(i / max(len(frame_data_list) - 1, 1) * 100),
                "bones": bones, "joints": joints,
                "angles": [{"name": k, "value": round(v, 1)} for k, v in fd.angles.items()]
            })
        return key_frames

    def _extract_angle_curves(self, frame_data_list):
        """提取角度曲线数据"""
        angle_curves = {}
        for aname in ['left_knee', 'right_knee', 'left_elbow', 'right_elbow',
                       'left_hip', 'right_hip', 'torso_from_vertical']:
            vals = [fd.angles.get(aname) for fd in frame_data_list if aname in fd.angles]
            if vals:
                if len(vals) > 100:
                    step = len(vals) // 100
                    vals = vals[::step][:100]
                angle_curves[aname] = [round(v, 1) for v in vals]
        return angle_curves