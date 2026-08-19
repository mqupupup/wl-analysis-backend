# backend/app/services/biomechanics_service.py
"""
生物力学分析服务 - 对外暴露的统一接口
"""

import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from collections import Counter

from .biomechanics import BiomechanicsEngine



class BiomechanicsService:
    """生物力学分析服务"""
    
    def __init__(self):
        self.engine = BiomechanicsEngine()
    
    def analyze_video(self, video_path: Path) -> Dict[str, Any]:
        """分析视频，返回完整结果"""
        analysis_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        print(f"\n{'='*70}")
        print(f"🎬 开始分析: {video_path.name}")
        print(f"📅 时间: {start_time}")
        print(f"🆔 ID: {analysis_id}")
        print(f"{'='*70}\n")
        
        try:
            result = self.engine.analyze(video_path)
            
            if not result.get('success'):
                return {
                    "analysis_id": analysis_id,
                    "success": False,
                    "error": result.get('error', '引擎分析失败')
                }
            v2_formatted_reps = result.get('v2_reps', [])
            exercise_type = result.get('exercise_type', 'Unknown')

            # ✅ 核心修复：生成结构化的 AI 智能反馈
            ai_feedback = self._generate_ai_feedback(
                result=result,
                v2_reps=v2_formatted_reps,
                exercise_type=exercise_type
            )

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print(f"\n{'='*70}")
            print(f"✅ 分析完成: {duration:.2f}秒")
            print(f"📊 动作: {result.get('exercise_type_zh', exercise_type)}")
            print(f"📊 次数: {result.get('rep_count', 0)}")
            print(f"📊 质量评分: {result.get('quality', {}).get('total_score', 'N/A')}")
            print(f"📊 置信度: {result.get('confidence', 'N/A')}%")
            print(f"{'='*70}\n")
            
            return {
                "analysis_id": analysis_id,
                "video_name": video_path.name,
                "processing_time_sec": round(duration, 2),
                "success": True,
                **result,
                "v2_reps": v2_formatted_reps,
                "ai_feedback": ai_feedback,  # ✅ 新增：结构化 AI 反馈
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "analysis_id": analysis_id,
                "success": False,
                "error": f"分析失败: {str(e)}"
            }

    def _generate_ai_feedback(
        self, 
        result: dict, 
        v2_reps: list, 
        exercise_type: str
    ) -> Dict[str, Any]:
        """将引擎返回的错误数据，转换成前端可直接展示的结构化 AI 反馈"""
        errors_raw = result.get('errors', {})
        total_score = result.get('quality', {}).get('total_score', 0)
        rep_count = result.get('rep_count', 0)
        fatigue = result.get('fatigue', {})
        exercise_zh = result.get('exercise_type_zh', exercise_type)
        
        # ========== 1. 解析错误数据 ==========
        # errors_raw 结构:
        #   total_count: int
        #   by_severity: {'medium': 11, 'high': 7}
        #   top_errors: [{'error_id', 'name_zh', 'severity', 'feedback'}, ...]
        
        total_error_count = errors_raw.get('total_count', 0) if isinstance(errors_raw, dict) else 0
        by_severity = errors_raw.get('by_severity', {}) if isinstance(errors_raw, dict) else {}
        top_errors = errors_raw.get('top_errors', []) if isinstance(errors_raw, dict) else []
        
        # 处理每条错误：修复模板变量 + 标准化
        error_details = []
        for err in top_errors:
            name_zh = err.get('name_zh', err.get('error_id', '未知错误'))
            severity = err.get('severity', 'medium')
            feedback = err.get('feedback', '')
            error_id = err.get('error_id', '')
            
            # ✅ 修复 {weaker_side} 模板变量未替换
            if '{weaker_side}' in feedback:
                # 尝试从 result 中找左右差值信息
                weaker_side = result.get('weaker_side', '')
                if not weaker_side:
                    # 如果引擎没提供，默认用"较弱"替代
                    weaker_side = '较弱'
                feedback = feedback.replace('{weaker_side}', weaker_side)
            
            # 统一 severity 命名（引擎用 high/medium，前端可能期望 severe/moderate）
            severity_map = {'high': 'severe', 'medium': 'moderate', 'low': 'minor'}
            severity_normalized = severity_map.get(severity, severity)
            
            # 根据 error_id 生成针对性改进建议
            suggestion = self._get_suggestion_for_error(error_id, feedback)
            
            error_details.append({
                "error_id": error_id,
                "name": name_zh,
                "severity": severity_normalized,
                "severity_raw": severity,
                "feedback": feedback,
                "suggestion": suggestion,
            })
        
        # 按严重程度排序（high 排前面）
        severity_order = {"critical": 0, "high": 1, "severe": 1, "medium": 2, "moderate": 2, "low": 3, "minor": 3}
        error_details.sort(key=lambda x: severity_order.get(x["severity_raw"], 9))
        
        # ========== 2. 分析 V2 评分的低分维度 ==========
        weak_dimensions = []
        if v2_reps:
            dim_scores = {}
            for rep in v2_reps:
                for dim in rep.get('dimensions', []):
                    dim_name = dim.get('name', dim.get('name_zh', ''))
                    score = dim.get('score', 100)
                    if dim_name:
                        if dim_name not in dim_scores:
                            dim_scores[dim_name] = []
                        dim_scores[dim_name].append(score)
            
            for dim_name, scores in dim_scores.items():
                avg_score = sum(scores) / len(scores) if scores else 100
                if avg_score < 70:
                    weak_dimensions.append({
                        "name": dim_name,
                        "avg_score": round(avg_score, 1),
                        "trend": "improving" if scores[-1] > scores[0] else "declining" if scores[-1] < scores[0] else "stable"
                    })
            weak_dimensions.sort(key=lambda x: x["avg_score"])
        
        # ========== 3. 生成总体评价 ==========
        if total_score >= 85:
            overall = "优秀"
            overall_emoji = "🏆"
            overall_desc = f"你的{exercise_zh}动作非常标准，继续保持！"
        elif total_score >= 70:
            overall = "良好"
            overall_emoji = "✅"
            overall_desc = f"你的{exercise_zh}整体不错，但有一些细节可以优化。"
        elif total_score >= 50:
            overall = "一般"
            overall_emoji = "⚠️"
            overall_desc = f"你的{exercise_zh}存在几个明显的问题，建议针对性改进。"
        else:
            overall = "需要改进"
            overall_emoji = "🔴"
            overall_desc = f"你的{exercise_zh}存在较多问题，建议降低重量，先纠正动作模式。"
        
        # ========== 4. 生成疲劳警告 ==========
        fatigue_warning = None
        fatigue_level = fatigue.get('level', fatigue.get('fatigue_level', ''))
        speed_loss = fatigue.get('speed_loss', 0)
        rir = fatigue.get('rir', fatigue.get('estimated_rir', None))
        
        if fatigue_level in ('high', 'severe') or (isinstance(speed_loss, (int, float)) and speed_loss > 0.3):
            speed_loss_pct = speed_loss * 100 if isinstance(speed_loss, float) and speed_loss < 1 else speed_loss
            fatigue_warning = {
                "level": fatigue_level,
                "speed_loss": f"{speed_loss_pct:.1f}%",
                "rir": rir,
                "message": f"⚠️ 检测到明显疲劳（速度下降 {speed_loss_pct:.0f}%），建议停止当前组或降低重量，避免受伤。"
            }
        
        # ========== 5. 生成优先改进建议（Top 3） ==========
        top_suggestions = []
        for err in error_details[:3]:
            top_suggestions.append({
                "problem": err["name"],
                "severity": err["severity_raw"],
                "feedback": err["feedback"],
                "suggestion": err["suggestion"],
            })
        
        # ========== 6. 构建总结文本 ==========
        summary_lines = [f"{overall_emoji} {overall_desc}", ""]
        
        if error_details:
            summary_lines.append("🔴 检测到的问题：")
            for i, err in enumerate(error_details, 1):
                sev_emoji = {"high": "🔴", "severe": "🔴", "medium": "🟡", "moderate": "🟡", "low": "🟢", "minor": "🟢"}.get(err["severity_raw"], "⚪")
                summary_lines.append(f"  {i}. {sev_emoji} {err['name']}")
                if err["feedback"]:
                    summary_lines.append(f"     {err['feedback']}")
            summary_lines.append("")
        
        if weak_dimensions:
            summary_lines.append("📊 评分较低的维度：")
            for dim in weak_dimensions[:3]:
                trend_icon = "📈" if dim["trend"] == "improving" else "📉" if dim["trend"] == "declining" else "➡️"
                summary_lines.append(f"  • {dim['name']}：{dim['avg_score']}分 {trend_icon}")
            summary_lines.append("")
        
        if fatigue_warning:
            summary_lines.append(fatigue_warning["message"])
            summary_lines.append("")
        
        if not error_details:
            summary_lines.append("✅ 未检测到明显动作错误，继续保持！")
        
        # ========== 7. 组装最终返回 ==========
        return {
            "overall": overall,
            "overall_emoji": overall_emoji,
            "overall_description": overall_desc,
            "total_score": total_score,
            "rep_count": rep_count,
            
            "errors_summary": {
                "total_count": total_error_count,
                "by_severity": by_severity,
                "details": error_details,
            },
            
            "weak_dimensions": weak_dimensions,
            "top_suggestions": top_suggestions,
            "fatigue_warning": fatigue_warning,
            "summary_text": "\n".join(summary_lines),
        }

    def _get_suggestion_for_error(self, error_id: str, feedback: str) -> str:
        """根据 error_id 返回针对性的改进建议"""
        suggestions = {
            "bench_bounce": "尝试「暂停卧推」训练法：杠铃触胸后停顿 1-2 秒再推起，培养底部控制力。可以先用 60% 1RM 的重量练习。",
            "bench_hip_lift": "调整起桥幅度：臀部始终贴紧凳面，只让上背部和肩膀接触凳面。收紧核心，脚掌稳固踩地。",
            "bench_asymmetric_push": "加入单侧训练：用哑铃卧推强化弱侧，每周 2-3 组。也可以在热身时用弹力带做单臂推举。",
            "bench_elbow_flare": "控制肘部角度：大臂与躯干夹角保持在 45-75°（箭头型而非 T 型），想象把杠铃「掰弯」来激活背阔肌。",
            "bench_incomplete_rom": "确保完整幅度：杠铃下放至轻触胸部，推起至手臂完全伸直。可以让训练伙伴帮忙观察。",
        }
        return suggestions.get(error_id, "")
    def _build_summary_text(
        self, 
        overall_desc: str, 
        errors: list, 
        weak_dims: list, 
        fatigue: dict,
        rep_count: int,
        exercise_zh: str
    ) -> str:
        """生成一段人类可读的总结文本"""
        lines = [overall_desc, ""]
        
        if errors:
            lines.append("🔴 主要问题：")
            for i, err in enumerate(errors[:3], 1):
                severity_emoji = {"severe": "🔴", "moderate": "🟡", "minor": "🟢"}.get(err["severity"], "⚪")
                lines.append(f"  {i}. {severity_emoji} {err['name']}（出现 {err['count']}/{rep_count} 次）")
                if err.get("suggestion"):
                    lines.append(f"     💡 {err['suggestion']}")
            lines.append("")
        
        if weak_dims:
            lines.append("📊 需要加强的维度：")
            for dim in weak_dims[:3]:
                lines.append(f"  • {dim['name']}：平均 {dim['avg_score']} 分（趋势：{'📈 改善中' if dim['trend'] == 'improving' else '📉 下降中' if dim['trend'] == 'declining' else '➡️ 稳定'}）")
            lines.append("")
        
        if fatigue:
            lines.append(fatigue["message"])
            lines.append("")
        
        if not errors and not weak_dims:
            lines.append("✅ 动作整体规范，继续保持！")
        
        return "\n".join(lines)

    def _get_exercise_knowledge(self, exercise_type: str) -> dict:
        """获取动作的先验知识库"""
        return {
            "meta": {"name_zh": exercise_type, "name_en": exercise_type},
            "layers": []
        }


# ================= 单例模式 =================
_service_instance = None

def get_service() -> BiomechanicsService:
    global _service_instance
    if _service_instance is None:
        _service_instance = BiomechanicsService()
    return _service_instance


def analyze(video_path: str | Path) -> dict:
    """对外暴露的便捷分析函数"""
    return get_service().analyze_video(Path(video_path))