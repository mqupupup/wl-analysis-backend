# app/services/v2_analysis_service.py (新建文件)

from app.services.biomechanics.exercise_specific_scorer_v2 import (
    ExerciseSpecificScorerV2, 
    RepScoreResult,
    format_v2_results_for_frontend,
)

def run_v2_analysis(reps, frame_data_list, knowledge):
    """V2 分析主流程"""
    scorer_v2 = ExerciseSpecificScorerV2(knowledge, fps=30.0)
    rep_results_v2 = scorer_v2.score_reps(reps, frame_data_list)
    
    return {
        "exercise": knowledge.get("meta", {}).get("name_zh", "未知动作"),
        "reps": format_v2_results_for_frontend(rep_results_v2)
    }