"""
知识库 JSON Schema 验证器
确保每个 exercise.json 结构完整、字段类型正确、规则可解析
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field


# ================================================================
# 验证结果
# ================================================================

@dataclass
class ValidationError:
    """单条验证错误"""
    path: str          # 错误所在的 JSON 路径，如 "errors[2].detect.condition"
    level: str         # "error" | "warning"
    message: str       # 错误描述
    suggestion: str = ""  # 修复建议

    def __str__(self):
        icon = "❌" if self.level == "error" else "⚠️"
        msg = f"{icon} [{self.path}] {self.message}"
        if self.suggestion:
            msg += f" → {self.suggestion}"
        return msg


@dataclass
class ValidationResult:
    """验证结果汇总"""
    exercise_id: str
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    def add_error(self, path: str, message: str, suggestion: str = ""):
        self.errors.append(ValidationError(path, "error", message, suggestion))

    def add_warning(self, path: str, message: str, suggestion: str = ""):
        self.warnings.append(ValidationError(path, "warning", message, suggestion))

    def summary(self) -> str:
        lines = []
        if self.is_valid:
            lines.append(f"✅ {self.exercise_id} 验证通过 ({self.warning_count} 个警告)")
        else:
            lines.append(f"❌ {self.exercise_id} 验证失败 ({self.error_count} 个错误, {self.warning_count} 个警告)")

        for e in self.errors:
            lines.append(f"  {e}")
        for w in self.warnings:
            lines.append(f"  {w}")
        return "\n".join(lines)


# ================================================================
# 允许的枚举值
# ================================================================

VALID_SEVERITIES = {"critical", "high", "medium", "low"}
VALID_MEASURE_METHODS = {
    "compare_y", "compare_x", "compare_x_at_phase", "compare_y_at_phase",
    "angle", "angle_at_phase", "angle_between", "angle_change_at_phase",
    "velocity_comparison", "midpoint_x_std", "midpoint_x_deviation",
    "wrist_midpoint_trajectory", "wrist_midpoint_x_trajectory",
    "wrist_y_relative_to_shoulder", "wrist_x_distance_to_hip",
    "wrist_x_relative_to_shoulder_at_lockout", "wrist_position_change",
    "elbow_position_stability", "elbow_y_change", "elbow_angle_diff",
    "elbow_angle_range",
    "hip_y_relative_to_shoulder_knee", "hip_knee_extension_at_phase",
    "torso_angle", "torso_straightness", "torso_angle_change",
    "torso_angle_at_phase", "torso_angle_oscillation", "torso_past_vertical",
    "torso_from_vertical", "lower_torso_angle",
    "compare_sides", "phase_duration", "phase_duration_ratio",
    "acceleration_at_phase", "acceleration_spike",
    "eccentric_duration_vs_concentric", "eccentric_vs_concentric_time",
    "min_elbow_angle", "max_elbow_angle_and_min_wrist_y",
    "knee_angle_change_at_start", "knee_distance_vs_ankle_distance",
    "hip_vs_shoulder_rise", "spine_flexion",
    "head_position_relative_to_bar", "bar_y_velocity",
}
VALID_PHASES = {
    "setup", "start", "eccentric", "bottom", "bottom_pause",
    "concentric", "concentric_first_half", "concentric_first_30pct",
    "mid_concentric", "mid_eccentric",
    "sticking_point", "lockout", "lockout_hold",
    "peak", "peak_contraction",
    "first_pull", "first_pull_start", "transition", "second_pull",
    "setup_to_concentric",
    "rest", "failed",
    "full_rep",  # 跨整个 rep
}


# ================================================================
# 主验证器
# ================================================================

class ExerciseKnowledgeValidator:
    """
    验证 exercise knowledge JSON 的结构完整性和语义正确性。
    
    用法:
        validator = ExerciseKnowledgeValidator()
        result = validator.validate_file("squat.json")
        print(result.summary())
    """

    def validate_file(self, filepath: str | Path) -> ValidationResult:
        """验证单个 JSON 文件"""
        filepath = Path(filepath)

        # 文件存在性
        if not filepath.exists():
            result = ValidationResult(exercise_id=filepath.stem)
            result.add_error("file", f"文件不存在: {filepath}")
            return result

        # JSON 解析
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            result = ValidationResult(exercise_id=filepath.stem)
            result.add_error("json", f"JSON 解析失败: {e}")
            return result

        return self.validate_data(data, exercise_id=filepath.stem)

    def validate_data(self, data: Dict[str, Any], exercise_id: str = "unknown") -> ValidationResult:
        """验证已解析的 JSON 数据"""
        result = ValidationResult(exercise_id=exercise_id)

        # 1. 顶层结构检查
        self._validate_top_level(data, result)

        # 2. meta 检查
        if "meta" in data:
            self._validate_meta(data["meta"], result)

        # 3. phases 检查
        if "phases" in data:
            self._validate_phases(data["phases"], result)

        # 4. key_points 检查
        if "key_points" in data:
            self._validate_key_points(data["key_points"], result)

        # 5. errors 检查
        if "errors" in data:
            self._validate_errors(data["errors"], result)

        # 6. rep_detection 检查
        if "rep_detection" in data:
            self._validate_rep_detection(data["rep_detection"], result)

        # 7. feedback_rules 检查
        if "feedback_rules" in data:
            self._validate_feedback_rules(data["feedback_rules"], result)

        # 8. 交叉引用检查
        self._validate_cross_references(data, result)

        return result

    def validate_all(self, directory: str | Path = None) -> List[ValidationResult]:
        """验证目录下所有 JSON 文件"""
        if directory is None:
            directory = Path(__file__).parent / "exercises"
        else:
            directory = Path(directory)

        results = []
        for f in sorted(directory.glob("*.json")):
            results.append(self.validate_file(f))

        return results

    # ================================================================
    # 顶层结构
    # ================================================================

    def _validate_top_level(self, data: Dict, result: ValidationResult):
        """验证顶层必需字段"""
        required_sections = ["meta", "phases", "key_points", "errors", "rep_detection", "feedback_rules"]
        for section in required_sections:
            if section not in data:
                result.add_error(
                    section,
                    f"缺少顶层字段 '{section}'",
                    f"请添加 {section} 配置"
                )

        # 检查是否有多余字段
        known_sections = set(required_sections)
        for key in data.keys():
            if key not in known_sections:
                result.add_warning(
                    key,
                    f"未知顶层字段 '{key}'",
                    "如果是自定义扩展字段可忽略"
                )

    # ================================================================
    # meta
    # ================================================================

    def _validate_meta(self, meta: Dict, result: ValidationResult):
        """验证 meta 部分"""
        prefix = "meta"

        if not isinstance(meta, dict):
            result.add_error(prefix, "meta 应为对象")
            return

        # 必需字段
        required = ["id", "name_en", "name_zh", "motion_chain", "pattern", "primary_joints"]
        for field in required:
            if field not in meta:
                result.add_error(f"{prefix}.{field}", f"缺少必需字段 '{field}'")

        # 类型检查
        if "id" in meta and not isinstance(meta["id"], str):
            result.add_error(f"{prefix}.id", "id 应为字符串")

        if "primary_joints" in meta:
            if not isinstance(meta["primary_joints"], list):
                result.add_error(f"{prefix}.primary_joints", "primary_joints 应为数组")
            elif len(meta["primary_joints"]) == 0:
                result.add_error(f"{prefix}.primary_joints", "primary_joints 不应为空")

        if "primary_muscles" in meta and not isinstance(meta["primary_muscles"], list):
            result.add_error(f"{prefix}.primary_muscles", "primary_muscles 应为数组")

        if "typical_rom" in meta:
            rom = meta["typical_rom"]
            if not isinstance(rom, dict):
                result.add_error(f"{prefix}.typical_rom", "typical_rom 应为对象")
            else:
                for joint, range_val in rom.items():
                    if not isinstance(range_val, dict):
                        result.add_error(f"{prefix}.typical_rom.{joint}", "应为 {min, max} 对象")
                    elif "min" not in range_val or "max" not in range_val:
                        result.add_error(f"{prefix}.typical_rom.{joint}", "缺少 min 或 max")
                    elif range_val.get("min", 0) >= range_val.get("max", 0):
                        result.add_error(
                            f"{prefix}.typical_rom.{joint}",
                            f"min ({range_val['min']}) 应小于 max ({range_val['max']})"
                        )

    # ================================================================
    # phases
    # ================================================================

    def _validate_phases(self, phases: Dict, result: ValidationResult):
        """验证 phases 状态机定义"""
        prefix = "phases"

        if not isinstance(phases, dict):
            result.add_error(prefix, "phases 应为对象")
            return

        # states
        if "states" not in phases:
            result.add_error(f"{prefix}.states", "缺少 states 定义")
        elif not isinstance(phases["states"], list):
            result.add_error(f"{prefix}.states", "states 应为数组")
        elif len(phases["states"]) < 3:
            result.add_error(f"{prefix}.states", "至少需要 3 个状态", "建议: setup, eccentric, lockout")

        states_set = set(phases.get("states", []))

        # transitions
        if "transitions" not in phases:
            result.add_error(f"{prefix}.transitions", "缺少 transitions 定义")
        elif not isinstance(phases["transitions"], list):
            result.add_error(f"{prefix}.transitions", "transitions 应为数组")
        else:
            for i, t in enumerate(phases["transitions"]):
                t_prefix = f"{prefix}.transitions[{i}]"
                if not isinstance(t, dict):
                    result.add_error(t_prefix, "每个 transition 应为对象")
                    continue

                # from / to
                for field in ["from", "to", "condition", "description"]:
                    if field not in t:
                        result.add_error(f"{t_prefix}.{field}", f"缺少 '{field}'")

                # from/to 必须在 states 中
                if "from" in t and t["from"] not in states_set:
                    result.add_error(
                        f"{t_prefix}.from",
                        f"状态 '{t['from']}' 不在 states 列表中",
                        f"可用状态: {sorted(states_set)}"
                    )
                if "to" in t and t["to"] not in states_set:
                    result.add_error(
                        f"{t_prefix}.to",
                        f"状态 '{t['to']}' 不在 states 列表中",
                        f"可用状态: {sorted(states_set)}"
                    )

                # condition 非空
                if "condition" in t and (not t["condition"] or not t["condition"].strip()):
                    result.add_error(f"{t_prefix}.condition", "condition 不应为空")

        # critical_frames (optional but recommended)
        if "critical_frames" not in phases:
            result.add_warning(f"{prefix}.critical_frames", "建议添加 critical_frames 定义")
        elif isinstance(phases["critical_frames"], dict):
            for frame_name, desc in phases["critical_frames"].items():
                if not isinstance(desc, str) or not desc.strip():
                    result.add_warning(
                        f"{prefix}.critical_frames.{frame_name}",
                        "描述应为非空字符串"
                    )

    # ================================================================
    # key_points
    # ================================================================

    def _validate_key_points(self, key_points: List, result: ValidationResult):
        """验证评分维度"""
        prefix = "key_points"

        if not isinstance(key_points, list):
            result.add_error(prefix, "key_points 应为数组")
            return

        if len(key_points) == 0:
            result.add_error(prefix, "key_points 不应为空")
            return

        if len(key_points) < 3:
            result.add_warning(prefix, "评分维度少于 3 个，可能不够全面")

        # 权重检查
        total_weight = 0.0
        seen_ids = set()

        for i, kp in enumerate(key_points):
            kp_prefix = f"{prefix}[{i}]"

            if not isinstance(kp, dict):
                result.add_error(kp_prefix, "每个 key_point 应为对象")
                continue

            # 必需字段
            for field in ["id", "name_zh", "name_en", "weight", "measure", "scoring", "score_map"]:
                if field not in kp:
                    result.add_error(f"{kp_prefix}.{field}", f"缺少 '{field}'")

            # id 唯一性
            kp_id = kp.get("id", "")
            if kp_id in seen_ids:
                result.add_error(f"{kp_prefix}.id", f"id '{kp_id}' 重复")
            seen_ids.add(kp_id)

            # 权重
            weight = kp.get("weight", 0)
            if not isinstance(weight, (int, float)):
                result.add_error(f"{kp_prefix}.weight", "weight 应为数字")
            elif weight < 0 or weight > 1:
                result.add_error(f"{kp_prefix}.weight", f"weight ({weight}) 应在 0-1 之间")
            else:
                total_weight += weight

            # measure 结构
            if "measure" in kp:
                self._validate_measure(kp["measure"], f"{kp_prefix}.measure", result)

            # scoring 结构
            if "scoring" in kp:
                if not isinstance(kp["scoring"], dict):
                    result.add_error(f"{kp_prefix}.scoring", "scoring 应为对象")
                else:
                    for level in ["excellent", "good", "poor"]:
                        if level not in kp["scoring"]:
                            result.add_warning(f"{kp_prefix}.scoring.{level}", f"建议添加 '{level}' 级别描述")

            # score_map 结构
            if "score_map" in kp:
                if not isinstance(kp["score_map"], dict):
                    result.add_error(f"{kp_prefix}.score_map", "score_map 应为对象")
                else:
                    for key, val in kp["score_map"].items():
                        if not isinstance(val, (int, float)):
                            result.add_error(f"{kp_prefix}.score_map.{key}", "分数应为数字")
                        elif val < 0 or val > 100:
                            result.add_warning(f"{kp_prefix}.score_map.{key}", f"分数 ({val}) 超出 0-100 范围")

        # 权重总和检查
        if abs(total_weight - 1.0) > 0.05:
            result.add_error(
                f"{prefix}.weights",
                f"所有 key_point 的 weight 之和 = {total_weight:.2f}，应为 1.0",
                "调整各维度权重使其总和为 1.0"
            )

    def _validate_measure(self, measure: Dict, prefix: str, result: ValidationResult):
        """验证 measure 子结构"""
        if not isinstance(measure, dict):
            result.add_error(prefix, "measure 应为对象")
            return

        if "method" not in measure:
            result.add_error(f"{prefix}.method", "缺少 method")
        elif measure["method"] not in VALID_MEASURE_METHODS:
            result.add_warning(
                f"{prefix}.method",
                f"未知测量方法 '{measure['method']}'",
                f"已知方法: {sorted(VALID_MEASURE_METHODS)}"
            )

        # at_phase 验证
        if "at_phase" in measure:
            phase_val = measure["at_phase"]
            if isinstance(phase_val, str):
                phases = [phase_val]
            elif isinstance(phase_val, list):
                phases = phase_val
            else:
                result.add_error(f"{prefix}.at_phase", "at_phase 应为字符串或字符串数组")
                phases = []

            for p in phases:
                if p not in VALID_PHASES:
                    result.add_warning(
                        f"{prefix}.at_phase",
                        f"未知阶段名 '{p}'",
                        f"已知阶段: {sorted(VALID_PHASES)}"
                    )

    # ================================================================
    # errors
    # ================================================================

    def _validate_errors(self, errors: List, result: ValidationResult):
        """验证错误检测库"""
        prefix = "errors"

        if not isinstance(errors, list):
            result.add_error(prefix, "errors 应为数组")
            return

        if len(errors) == 0:
            result.add_warning(prefix, "没有定义任何错误检测规则")

        seen_ids = set()

        for i, err in enumerate(errors):
            err_prefix = f"{prefix}[{i}]"

            if not isinstance(err, dict):
                result.add_error(err_prefix, "每个 error 应为对象")
                continue

            # 必需字段
            for field in ["id", "name_zh", "name_en", "severity", "detect", "feedback_zh"]:
                if field not in err:
                    result.add_error(f"{err_prefix}.{field}", f"缺少 '{field}'")

            # id 唯一性
            err_id = err.get("id", "")
            if err_id in seen_ids:
                result.add_error(f"{err_prefix}.id", f"id '{err_id}' 重复")
            seen_ids.add(err_id)

            # id 命名规范
            if err_id and "_" not in err_id:
                result.add_warning(
                    f"{err_prefix}.id",
                    f"id '{err_id}' 建议用 '动作_错误' 格式命名",
                    "如: squat_knee_valgus"
                )

            # severity
            if "severity" in err:
                if err["severity"] not in VALID_SEVERITIES:
                    result.add_error(
                        f"{err_prefix}.severity",
                        f"无效的 severity '{err['severity']}'",
                        f"允许值: {sorted(VALID_SEVERITIES)}"
                    )

            # detect 结构
            if "detect" in err:
                self._validate_detect(err["detect"], f"{err_prefix}.detect", result)

            # severity_levels
            if "severity_levels" in err:
                if not isinstance(err["severity_levels"], dict):
                    result.add_error(f"{err_prefix}.severity_levels", "应为对象")
                else:
                    for level_name, level_data in err["severity_levels"].items():
                        if level_name not in ("minor", "moderate", "severe"):
                            result.add_warning(
                                f"{err_prefix}.severity_levels.{level_name}",
                                f"未知级别 '{level_name}'，建议用 minor/moderate/severe"
                            )
                        if isinstance(level_data, dict):
                            if "threshold" not in level_data:
                                result.add_error(
                                    f"{err_prefix}.severity_levels.{level_name}.threshold",
                                    "缺少 threshold"
                                )
                            if "score_penalty" not in level_data:
                                result.add_error(
                                    f"{err_prefix}.severity_levels.{level_name}.score_penalty",
                                    "缺少 score_penalty"
                                )
                            elif not isinstance(level_data["score_penalty"], (int, float)):
                                result.add_error(
                                    f"{err_prefix}.severity_levels.{level_name}.score_penalty",
                                    "score_penalty 应为数字"
                                )
                            elif level_data["score_penalty"] > 0:
                                result.add_warning(
                                    f"{err_prefix}.severity_levels.{level_name}.score_penalty",
                                    f"score_penalty ({level_data['score_penalty']}) 应为负数",
                                    "改为负值如 -10"
                                )

            # injury_risk
            if "injury_risk" in err:
                if not isinstance(err["injury_risk"], list):
                    result.add_error(f"{err_prefix}.injury_risk", "应为数组")
                elif len(err["injury_risk"]) == 0:
                    result.add_warning(f"{err_prefix}.injury_risk", "injury_risk 为空")

            # fix_exercises
            if "fix_exercises" in err:
                if not isinstance(err["fix_exercises"], list):
                    result.add_error(f"{err_prefix}.fix_exercises", "应为数组")
                elif len(err["fix_exercises"]) == 0:
                    result.add_warning(f"{err_prefix}.fix_exercises", "fix_exercises 为空")

            # feedback 文案检查
            if "feedback_zh" in err:
                fb = err["feedback_zh"]
                if not isinstance(fb, str):
                    result.add_error(f"{err_prefix}.feedback_zh", "应为字符串")
                elif len(fb) < 10:
                    result.add_warning(f"{err_prefix}.feedback_zh", "反馈文案过短，可能不够具体")

            if "feedback_en" in err:
                fb = err["feedback_en"]
                if not isinstance(fb, str):
                    result.add_error(f"{err_prefix}.feedback_en", "应为字符串")

    def _validate_detect(self, detect: Dict, prefix: str, result: ValidationResult):
        """验证 detect 子结构"""
        if not isinstance(detect, dict):
            result.add_error(prefix, "detect 应为对象")
            return

        if "method" not in detect:
            result.add_error(f"{prefix}.method", "缺少 method")

        if "condition" not in detect:
            result.add_error(f"{prefix}.condition", "缺少 condition")
        elif not isinstance(detect["condition"], str):
            result.add_error(f"{prefix}.condition", "condition 应为字符串")

        if "phase" not in detect:
            result.add_warning(f"{prefix}.phase", "建议指定检测阶段 (phase)")
        else:
            phase_val = detect["phase"]
            if isinstance(phase_val, str):
                phases = [phase_val]
            elif isinstance(phase_val, list):
                phases = phase_val
            else:
                result.add_error(f"{prefix}.phase", "phase 应为字符串或数组")
                phases = []

            for p in phases:
                if p not in VALID_PHASES:
                    result.add_warning(
                        f"{prefix}.phase",
                        f"未知阶段 '{p}'",
                        f"已知: {sorted(VALID_PHASES)}"
                    )

        if "min_frames" in detect:
            if not isinstance(detect["min_frames"], int):
                result.add_error(f"{prefix}.min_frames", "min_frames 应为整数")
            elif detect["min_frames"] < 1:
                result.add_error(f"{prefix}.min_frames", "min_frames 应 ≥ 1")

    # ================================================================
    # rep_detection
    # ================================================================

    def _validate_rep_detection(self, rep_det: Dict, result: ValidationResult):
        """验证 rep 检测规则"""
        prefix = "rep_detection"

        if not isinstance(rep_det, dict):
            result.add_error(prefix, "rep_detection 应为对象")
            return

        required = ["primary_angle", "min_rom", "min_duration", "max_duration"]
        for field in required:
            if field not in rep_det:
                result.add_error(f"{prefix}.{field}", f"缺少 '{field}'")

        # 数值范围检查
        if "min_rom" in rep_det:
            v = rep_det["min_rom"]
            if not isinstance(v, (int, float)) or v <= 0:
                result.add_error(f"{prefix}.min_rom", "min_rom 应为正数")

        if "min_duration" in rep_det:
            v = rep_det["min_duration"]
            if not isinstance(v, (int, float)) or v <= 0:
                result.add_error(f"{prefix}.min_duration", "min_duration 应为正数（秒）")

        if "max_duration" in rep_det:
            v = rep_det["max_duration"]
            if not isinstance(v, (int, float)) or v <= 0:
                result.add_error(f"{prefix}.max_duration", "max_duration 应为正数（秒）")

        if "min_duration" in rep_det and "max_duration" in rep_det:
            if rep_det["min_duration"] >= rep_det["max_duration"]:
                result.add_error(
                    prefix,
                    f"min_duration ({rep_det['min_duration']}) 应小于 max_duration ({rep_det['max_duration']})"
                )

        # peak/valley 类型
        if "peak_type" in rep_det:
            if rep_det["peak_type"] not in ("maxima", "minima"):
                result.add_error(f"{prefix}.peak_type", "peak_type 应为 'maxima' 或 'minima'")

        if "valley_type" in rep_det:
            if rep_det["valley_type"] not in ("maxima", "minima"):
                result.add_error(f"{prefix}.valley_type", "valley_type 应为 'maxima' 或 'minima'")

    # ================================================================
    # feedback_rules
    # ================================================================

    def _validate_feedback_rules(self, rules: Dict, result: ValidationResult):
        """验证反馈规则"""
        prefix = "feedback_rules"

        if not isinstance(rules, dict):
            result.add_error(prefix, "feedback_rules 应为对象")
            return

        expected_levels = ["excellent", "good", "needs_work", "poor"]
        for level in expected_levels:
            if level not in rules:
                result.add_warning(f"{prefix}.{level}", f"建议添加 '{level}' 级别反馈")
            else:
                rule = rules[level]
                if not isinstance(rule, dict):
                    result.add_error(f"{prefix}.{level}", "应为对象")
                    continue

                if "condition" not in rule:
                    result.add_error(f"{prefix}.{level}.condition", "缺少 condition")

                if "zh" not in rule:
                    result.add_error(f"{prefix}.{level}.zh", "缺少中文反馈 (zh)")
                elif not isinstance(rule["zh"], str) or len(rule["zh"]) < 5:
                    result.add_warning(f"{prefix}.{level}.zh", "中文反馈过短")

                # 检查模板变量
                zh_text = rule.get("zh", "")
                if "{" in zh_text:
                    # 提取模板变量
                    import re
                    vars_found = set(re.findall(r'\{(\w+)\}', zh_text))
                    known_vars = {"rep_count", "total_score", "top_error", "second_error",
                                  "exercise", "angle", "severity", "weaker_side"}
                    for v in vars_found:
                        if v not in known_vars:
                            result.add_warning(
                                f"{prefix}.{level}.zh",
                                f"未知模板变量 '{{{v}}}'",
                                f"已知变量: {sorted(known_vars)}"
                            )

    # ================================================================
    # 交叉引用检查
    # ================================================================

    def _validate_cross_references(self, data: Dict, result: ValidationResult):
        """验证各部分之间的引用一致性"""

        # 1. phases.states 与 errors[].detect.phase 的一致性
        phases_states = set()
        if "phases" in data and isinstance(data["phases"], dict):
            phases_states = set(data["phases"].get("states", []))

        if "errors" in data and isinstance(data["errors"], list):
            for i, err in enumerate(data["errors"]):
                detect = err.get("detect", {})
                phase_val = detect.get("phase")
                if phase_val:
                    phases_list = [phase_val] if isinstance(phase_val, str) else phase_val
                    for p in phases_list:
                        if phases_list and p not in phases_states and p not in VALID_PHASES:
                            result.add_error(
                                f"errors[{i}].detect.phase",
                                f"阶段 '{p}' 未在 phases.states 中定义",
                                f"可用: {sorted(phases_states)}"
                            )

        # 2. key_points 的 at_phase 与 phases.states 一致性
        if "key_points" in data and isinstance(data["key_points"], list):
            for i, kp in enumerate(data["key_points"]):
                measure = kp.get("measure", {})
                phase_val = measure.get("at_phase")
                if phase_val:
                    phases_list = [phase_val] if isinstance(phase_val, str) else phase_val
                    for p in phases_list:
                        if p not in phases_states and p not in VALID_PHASES:
                            result.add_warning(
                                f"key_points[{i}].measure.at_phase",
                                f"阶段 '{p}' 未在 phases.states 中定义"
                            )

        # 3. rep_detection.primary_angle 应在 typical_rom 中有对应
        if "rep_detection" in data and "meta" in data:
            primary = data["rep_detection"].get("primary_angle", "")
            rom = data["meta"].get("typical_rom", {})
            # 尝试匹配
            matched = False
            for joint in rom.keys():
                if joint in primary or primary.replace("_avg", "").replace("_angle", "") in joint:
                    matched = True
                    break
            if not matched and rom:
                result.add_warning(
                    "rep_detection.primary_angle",
                    f"primary_angle '{primary}' 在 meta.typical_rom 中无直接对应",
                    f"typical_rom 定义了: {sorted(rom.keys())}"
                )

        # 4. errors 的 severity 分布检查
        if "errors" in data and isinstance(data["errors"], list):
            severity_counts = {}
            for err in data["errors"]:
                sev = err.get("severity", "unknown")
                severity_counts[sev] = severity_counts.get(sev, 0) + 1

            if "critical" in severity_counts and severity_counts["critical"] > 2:
                result.add_warning(
                    "errors",
                    f"有 {severity_counts['critical']} 个 critical 级别错误，建议只保留最危险的 1-2 个"
                )

            if len(data["errors"]) > 10:
                result.add_warning(
                    "errors",
                    f"定义了 {len(data['errors'])} 个错误检测，数量较多",
                    "建议优先保留高频且有明确修复方案的错误"
                )


# ================================================================
# CLI 入口
# ================================================================

def main():
    """命令行验证所有知识库文件"""
    validator = ExerciseKnowledgeValidator()

    knowledge_dir = Path(__file__).parent / "exercises"
    if not knowledge_dir.exists():
        print(f"❌ 知识库目录不存在: {knowledge_dir}")
        return

    json_files = sorted(knowledge_dir.glob("*.json"))
    if not json_files:
        print(f"❌ 知识库目录为空: {knowledge_dir}")
        return

    print(f"📋 验证知识库目录: {knowledge_dir}")
    print(f"📁 找到 {len(json_files)} 个 JSON 文件\n")

    total_errors = 0
    total_warnings = 0

    for f in json_files:
        result = validator.validate_file(f)
        total_errors += result.error_count
        total_warnings += result.warning_count
        print(result.summary())
        print()

    print(f"{'=' * 60}")
    print(f"📊 汇总: {len(json_files)} 个文件, {total_errors} 个错误, {total_warnings} 个警告")

    if total_errors == 0:
        print("✅ 所有知识库文件验证通过！")
    else:
        print("❌ 存在验证错误，请修复后重新运行")

    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    exit(main())
    
# py -3.11 -m app.services.biomechanics.knowledge.validator