import numpy as np
import math
from typing import Dict, Any, Optional


def extract_landmarks_safe(detection_result):
    if not detection_result:
        return None

    lms = None
    if hasattr(detection_result, 'pose_landmarks') and detection_result.pose_landmarks:
        lms = detection_result.pose_landmarks[0]
    if not lms:
        return None

    landmarks_dict = {}
    for i, lm in enumerate(lms):
        visibility = getattr(lm, 'visibility', 1.0)
        presence = getattr(lm, 'presence', 1.0)
        # ✅ 修复：阈值从 0.3~0.5 降至 0.1，防止边缘关键点被误杀导致整帧丢弃
        if visibility > 0.1 and presence > 0.1:
            landmarks_dict[i] = {
                'x': float(lm.x), 'y': float(lm.y), 'z': float(lm.z),
                'visibility': float(visibility), 'presence': float(presence)
            }

    # ✅ 修复：不再强制 >=10 个关键点，有任意有效点即返回
    return landmarks_dict if len(landmarks_dict) > 0 else None


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def compute_frame_data(landmarks_dict: Dict[int, Dict], frame_idx: int,
                       frame_width: int, frame_height: int,
                       timestamp: float = 0.0):
    fw, fh = frame_width, frame_height

    def get_px(idx):
        if idx not in landmarks_dict:
            return None
        lm = landmarks_dict[idx]
        return [lm['x'] * fw, lm['y'] * fh]

    angles = {}
    positions = {}

    key_indices = {
        'left_shoulder': 11, 'right_shoulder': 12,
        'left_elbow': 13, 'right_elbow': 14,
        'left_wrist': 15, 'right_wrist': 16,
        'left_hip': 23, 'right_hip': 24,
        'left_knee': 25, 'right_knee': 26,
        'left_ankle': 27, 'right_ankle': 28,
    }

    for name, idx in key_indices.items():
        pt = get_px(idx)
        if pt:
            positions[name] = pt

    angle_defs = [
        ('left_elbow', 11, 13, 15), ('right_elbow', 12, 14, 16),
        ('left_knee', 23, 25, 27), ('right_knee', 24, 26, 28),
        ('left_hip', 11, 23, 25), ('right_hip', 12, 24, 26),
        ('left_shoulder_flex', 23, 11, 13), ('right_shoulder_flex', 24, 12, 14),
    ]

    for name, a_idx, b_idx, c_idx in angle_defs:
        pa, pb, pc = get_px(a_idx), get_px(b_idx), get_px(c_idx)
        if pa and pb and pc:
            angles[name] = calculate_angle(pa, pb, pc)

    ls = get_px(11); rs = get_px(12)
    lh = get_px(23); rh = get_px(24)
    if ls and rs and lh and rh:
        mid_shoulder = [(ls[0]+rs[0])/2, (ls[1]+rs[1])/2]
        mid_hip = [(lh[0]+rh[0])/2, (lh[1]+rh[1])/2]
        dx = mid_shoulder[0] - mid_hip[0]
        dy = mid_hip[1] - mid_shoulder[1]
        angles['torso_from_vertical'] = math.degrees(math.atan2(abs(dx), max(dy, 1)))

    if ls and get_px(13) and lh:
        angles['left_shoulder_abduction'] = calculate_angle(lh, ls, get_px(13))
    if rs and get_px(14) and rh:
        angles['right_shoulder_abduction'] = calculate_angle(rh, rs, get_px(14))

    for side, hip_idx, knee_idx, ankle_idx in [('left', 23, 25, 27), ('right', 24, 26, 28)]:
        hip_pt = get_px(hip_idx)
        knee_pt = get_px(knee_idx)
        ankle_pt = get_px(ankle_idx)
        if hip_pt and knee_pt and ankle_pt:
            mid_x = (hip_pt[0] + ankle_pt[0]) / 2
            angles[f'{side}_knee_valgus_offset'] = knee_pt[0] - mid_x

    lw = get_px(15); rw = get_px(16)
    if lw and rw and lh and rh:
        bar_x = (lw[0] + rw[0]) / 2
        body_center_x = (lh[0] + rh[0]) / 2
        angles['bar_body_distance'] = abs(bar_x - body_center_x)

    from app.domain.models import FramePoseData
    return FramePoseData(
        frame_idx=frame_idx,
        landmarks=landmarks_dict,
        angles=angles,
        positions=positions,
        timestamp_sec=timestamp
    )