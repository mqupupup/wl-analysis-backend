# app/engine/classifier.py
import numpy as np
from typing import List, Tuple
from app.domain.models import FramePoseData
from app.domain.enums import MovementPattern

def classify_movement_pattern(frame_data_list: List[FramePoseData], frame_height: int) -> Tuple[MovementPattern, float]:
    """
    V10 三状态门控分类器
    基于躯干角度、髋/膝/肘关节角度，识别动作模式。
    保证返回 (MovementPattern, confidence) 元组。
    """
    if not frame_data_list or len(frame_data_list) < 10:
        return MovementPattern.UNKNOWN, 0.0

    # 1. 提取关键角度序列
    torso_angles = [fd.angles.get('torso_from_vertical', 0) for fd in frame_data_list]
    hip_angles = [min(fd.angles.get('left_hip', 180), fd.angles.get('right_hip', 180)) for fd in frame_data_list]
    knee_angles = [min(fd.angles.get('left_knee', 180), fd.angles.get('right_knee', 180)) for fd in frame_data_list]
    elbow_angles = [min(fd.angles.get('left_elbow', 180), fd.angles.get('right_elbow', 180)) for fd in frame_data_list]
    shoulder_flex_angles = [max(fd.angles.get('left_shoulder_flex', 0), fd.angles.get('right_shoulder_flex', 0)) for fd in frame_data_list]

    # 2. 计算特征统计量 (最小值/最大值/活动范围)
    min_torso = min(torso_angles) if torso_angles else 0
    max_torso = max(torso_angles) if torso_angles else 0
    
    min_hip = min(hip_angles) if hip_angles else 180
    min_knee = min(knee_angles) if knee_angles else 180
    min_elbow = min(elbow_angles) if elbow_angles else 180
    max_shoulder_flex = max(shoulder_flex_angles) if shoulder_flex_angles else 0

    hip_rom = max(hip_angles) - min_hip if hip_angles else 0
    knee_rom = max(knee_angles) - min_knee if knee_angles else 0
    elbow_rom = max(elbow_angles) - min_elbow if elbow_angles else 0

    # 3. 门控分类逻辑 (V10)
    scores = {
        MovementPattern.LOWER_BODY_SQUAT: 0.0,
        MovementPattern.LOWER_BODY_HINGE: 0.0,
        MovementPattern.UPPER_BODY_HORIZONTAL_PUSH: 0.0,
        MovementPattern.UPPER_BODY_VERTICAL_PUSH: 0.0,
    }

    # 深蹲特征：膝盖弯曲大，躯干相对直立
    if min_knee < 120 and knee_rom > 30:
        scores[MovementPattern.LOWER_BODY_SQUAT] += 30
        if max_torso < 45:
            scores[MovementPattern.LOWER_BODY_SQUAT] += 20

    # 硬拉特征：髋关节弯曲大，躯干前倾大
    if min_hip < 120 and hip_rom > 30:
        scores[MovementPattern.LOWER_BODY_HINGE] += 30
        if max_torso > 40:
            scores[MovementPattern.LOWER_BODY_HINGE] += 20

    # 卧推特征：肘关节弯曲大 (视觉上门控较难，主要靠肘关节活动度)
    if min_elbow < 120 and elbow_rom > 30:
        scores[MovementPattern.UPPER_BODY_HORIZONTAL_PUSH] += 40

    # 过头推举特征：肩关节屈曲大，肘关节伸直
    if max_shoulder_flex > 140:
        scores[MovementPattern.UPPER_BODY_VERTICAL_PUSH] += 40

    # 4. 选出得分最高的模式
    best_pattern = max(scores, key=scores.get)
    best_score = scores[best_pattern]

    # 5. 置信度计算与兜底
    if best_score < 30:
        return MovementPattern.UNKNOWN, 0.0

    confidence = min(100.0, best_score * 1.5)
    return best_pattern, confidence