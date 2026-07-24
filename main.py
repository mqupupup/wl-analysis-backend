from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Tuple
import uvicorn
import cv2
import numpy as np
import os
import uuid
import shutil
import math
from datetime import datetime
import ffmpeg
from pathlib import Path
import time
import base64
import json

# ================== MediaPipe 初始化 ==================
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    model_path = Path(__file__).parent / "pose_landmarker.task"
    if not model_path.exists():
        print("💡 正在下载 MediaPipe 模型...")
        import urllib.request, ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
            model_path
        )

    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )
    pose_landmarker = vision.PoseLandmarker.create_from_options(options)
    MEDIAPIPE_AVAILABLE = True
    print(f"✅ MediaPipe {mp.__version__} (IMAGE模式) 加载成功")
except Exception as e:
    print(f"❌ MediaPipe 初始化失败: {e}")
    MEDIAPIPE_AVAILABLE = False
    pose_landmarker = None

# ================== YOLO 初始化 (仅作辅助，非核心) ==================
try:
    from ultralytics import YOLO
    yolo_path = Path(__file__).parent / "yolov8n.pt"
    if yolo_path.exists():
        barbell_model = YOLO(str(yolo_path))
        YOLO_AVAILABLE = True
        print("✅ YOLOv8 加载成功 (仅作辅助参考)")
    else:
        YOLO_AVAILABLE = False
        print("⚠️ YOLOv8 模型不存在 (不影响核心分析)")
except Exception as e:
    YOLO_AVAILABLE = False
    print(f"⚠️ YOLOv8 加载失败: {e}")


# ================== 中文名映射工具 ==================
EXERCISE_ZH_MAP = {
    "Squat": "深蹲",
    "Bench Press": "卧推",
    "Deadlift": "硬拉",
    "Overhead Press": "过头推举",
    "Unknown": "未知动作"
}


class BiomechanicsV70Classifier:

    @staticmethod
    def calculate_angle(a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba, bc = a - b, c - b
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    @staticmethod
    def extract_landmarks_safe(detection_result):
        if not detection_result or not detection_result.pose_landmarks:
            return None
        lms = detection_result.pose_landmarks[0]
        landmarks_dict = {}
        for i, l in enumerate(lms):
            visibility = getattr(l, 'visibility', 1.0)
            presence = getattr(l, 'presence', 1.0)
            if visibility > 0.3 and presence > 0.3:
                landmarks_dict[i] = type('Landmark', (), {
                    'x': l.x, 'y': l.y, 'z': l.z,
                    'visibility': visibility, 'presence': presence
                })()
        return landmarks_dict if len(landmarks_dict) >= 10 else None

    @staticmethod
    def _trunk_angle_from_landmarks(lm_dict, frame_height, frame_width):
        ls = lm_dict.get(11); rs = lm_dict.get(12)
        lh = lm_dict.get(23); rh = lm_dict.get(24)
        if not all([ls, rs, lh, rh]): return None
        sx = ((ls.x + rs.x) / 2) * frame_width
        sy = ((ls.y + rs.y) / 2) * frame_height
        hx = ((lh.x + rh.x) / 2) * frame_width
        hy = ((lh.y + rh.y) / 2) * frame_height
        dx = hx - sx; dy = hy - sy
        length = math.sqrt(dx**2 + dy**2)
        if length < 10: return None
        cos_a = max(-1.0, min(1.0, dy / length))
        return math.degrees(math.acos(cos_a))

    @staticmethod
    def calculate_trunk_orientation_multiframe(landmarks_list, frame_height, frame_width):
        angles = []; shoulder_hip_diffs = []
        for lm_dict in landmarks_list:
            if lm_dict is None: continue
            angle = BiomechanicsV70Classifier._trunk_angle_from_landmarks(lm_dict, frame_height, frame_width)
            if angle is not None: angles.append(angle)
            ls = lm_dict.get(11); rs = lm_dict.get(12)
            lh = lm_dict.get(23); rh = lm_dict.get(24)
            if all([ls, rs, lh, rh]):
                sy = (ls.y + rs.y) / 2 * frame_height
                hy = (lh.y + rh.y) / 2 * frame_height
                shoulder_hip_diffs.append(abs(sy - hy))
        if not angles: return 'vertical'
        median_angle = np.median(angles)
        median_sh_diff = np.median(shoulder_hip_diffs) if shoulder_hip_diffs else frame_height * 0.3
        print(f"  📐 躯干多帧中位数角度: {median_angle:.1f}°, 肩髋Y差中位数: {median_sh_diff:.0f}px")
        is_horizontal_by_angle = median_angle > 55
        is_vertical_by_angle = median_angle < 35
        is_horizontal_by_sh_diff = median_sh_diff < frame_height * 0.15
        is_vertical_by_sh_diff = median_sh_diff > frame_height * 0.20
        knee_below_hip = False
        last_lm = next((lm for lm in reversed(landmarks_list) if lm is not None), None)
        if last_lm:
            lk = last_lm.get(25); rk = last_lm.get(26)
            lh = last_lm.get(23); rh = last_lm.get(24)
            if all([lk, rk, lh, rh]):
                knee_y = (lk.y + rk.y) / 2 * frame_height
                hip_y = (lh.y + rh.y) / 2 * frame_height
                knee_below_hip = knee_y > hip_y + 20
        horizontal_score = 0; vertical_score = 0
        if is_horizontal_by_angle: horizontal_score += 3
        elif is_vertical_by_angle: vertical_score += 3
        else:
            if median_angle > 45: horizontal_score += 1
            else: vertical_score += 1
        if is_horizontal_by_sh_diff: horizontal_score += 2
        elif is_vertical_by_sh_diff: vertical_score += 2
        if knee_below_hip: vertical_score += 1
        else: horizontal_score += 1
        print(f"  📊 朝向评分: horizontal={horizontal_score}, vertical={vertical_score}")
        return 'horizontal' if horizontal_score > vertical_score else 'vertical'

    @staticmethod
    def calculate_trunk_orientation(landmarks_dict, frame_width, frame_height):
        angle = BiomechanicsV70Classifier._trunk_angle_from_landmarks(landmarks_dict, frame_height, frame_width)
        if angle is None: return 'vertical'
        return 'horizontal' if angle > 55 else 'vertical'

    @staticmethod
    def infer_barbell_from_wrists(landmarks_dict, frame_width, frame_height):
        if not landmarks_dict: return None
        lw = landmarks_dict.get(15); rw = landmarks_dict.get(16)
        ls = landmarks_dict.get(11); rs = landmarks_dict.get(12)
        lh = landmarks_dict.get(23); rh = landmarks_dict.get(24)
        la = landmarks_dict.get(27); ra = landmarks_dict.get(28)
        if not lw or not rw or not ls or not lh: return None
        wrist_y = ((lw.y + rw.y) / 2) * frame_height
        shoulder_y = ((ls.y + (rs.y if rs else ls.y)) / 2) * frame_height
        hip_y = ((lh.y + (rh.y if rh else lh.y)) / 2) * frame_height
        sh_center_y = (shoulder_y + hip_y) / 2
        sh_diff = abs(shoulder_y - hip_y)
        ankle_y = None
        if la: ankle_y = ((la.y + (ra.y if ra else la.y)) / 2) * frame_height
        tolerance = frame_height * 0.06
        ground_tolerance = frame_height * 0.10
        if ankle_y and abs(wrist_y - ankle_y) < ground_tolerance: return 'on_ground'
        if wrist_y > frame_height * 0.78: return 'on_ground'
        if wrist_y < shoulder_y - tolerance: return 'above_shoulder'
        is_lying = sh_diff < frame_height * 0.18
        wrist_above_body = wrist_y < sh_center_y - tolerance
        if is_lying and wrist_above_body: return 'above_chest'
        if wrist_above_body and wrist_y < shoulder_y - tolerance: return 'above_shoulder'
        return None

    @staticmethod
    def vote_barbell_position(all_barbell_positions, landmarks_history, frame_width, frame_height):
        position_counts = {'above_shoulder': 0, 'on_ground': 0, 'above_chest': 0, 'unknown': 0}
        for i, barbell in enumerate(all_barbell_positions):
            lm = landmarks_history[i] if i < len(landmarks_history) else None
            detected = False
            if barbell is not None:
                x, y, _ = barbell
                if lm:
                    pos_type = BiomechanicsV70Classifier._check_barbell_vs_body((x, y), lm, frame_width, frame_height)
                    if pos_type:
                        position_counts[pos_type] += 2; detected = True
                    else: position_counts['unknown'] += 1
            if lm and not detected:
                wrist_pos = BiomechanicsV70Classifier.infer_barbell_from_wrists(lm, frame_width, frame_height)
                if wrist_pos: position_counts[wrist_pos] += 1
                else: position_counts['unknown'] += 1
            elif not detected: position_counts['unknown'] += 1
        print(f"  🗳️ 杠铃位置投票结果 (双通道): {position_counts}")
        total_valid = sum(position_counts.values())
        if total_valid == 0: return None
        best_type = max(position_counts, key=position_counts.get)
        best_count = position_counts[best_type]
        ratio = best_count / total_valid
        if ratio >= 0.3 and best_type != 'unknown':
            print(f"  🎯 杠铃位置: {best_type} (得票率 {ratio:.0%})")
            return best_type
        print(f"  ⚠️ 杠铃位置投票未达阈值，返回 None")
        return None

    @staticmethod
    def _check_barbell_vs_body(barbell_pos, landmarks_dict, frame_width, frame_height):
        if not barbell_pos or not landmarks_dict: return None
        bx, by = barbell_pos[0], barbell_pos[1]
        ls = landmarks_dict.get(11); rs = landmarks_dict.get(12)
        lh = landmarks_dict.get(23); rh = landmarks_dict.get(24)
        la = landmarks_dict.get(27)
        if not ls or not lh: return None
        sy = ((ls.y + (rs.y if rs else ls.y)) / 2) * frame_height
        hy = ((lh.y + (rh.y if rh else lh.y)) / 2) * frame_height
        sh_center = (sy + hy) / 2; sh_diff = abs(sy - hy)
        tolerance = frame_height * 0.06; ground_tol = frame_height * 0.10
        if la:
            ay = ((la.y + (landmarks_dict.get(28).y if landmarks_dict.get(28) else la.y)) / 2) * frame_height
            if abs(by - ay) < ground_tol: return 'on_ground'
        if by > frame_height * 0.78: return 'on_ground'
        if by < sy - tolerance: return 'above_shoulder'
        if sh_diff < frame_height * 0.18 and by < sh_center - tolerance: return 'above_chest'
        return None

    @staticmethod
    def compute_joint_angles(landmarks_dict, frame_width, frame_height):
        def get_point(idx):
            if idx not in landmarks_dict: return None
            lm = landmarks_dict[idx]
            return [lm.x * frame_width, lm.y * frame_height]
        angles = {}
        pts = {k: get_point(k) for k in [11,12,13,14,15,16,23,24,25,26,27,28,29,30]}
        if pts[11] and pts[13] and pts[15]:
            try: angles['left_elbow'] = BiomechanicsV70Classifier.calculate_angle(pts[11], pts[13], pts[15])
            except: pass
        if pts[12] and pts[14] and pts[16]:
            try: angles['right_elbow'] = BiomechanicsV70Classifier.calculate_angle(pts[12], pts[14], pts[16])
            except: pass
        if pts[23] and pts[25] and pts[27]:
            try: angles['left_knee'] = BiomechanicsV70Classifier.calculate_angle(pts[23], pts[25], pts[27])
            except: pass
        if pts[24] and pts[26] and pts[28]:
            try: angles['right_knee'] = BiomechanicsV70Classifier.calculate_angle(pts[24], pts[26], pts[28])
            except: pass
        if pts[11] and pts[23] and pts[25]:
            try: angles['left_hip'] = BiomechanicsV70Classifier.calculate_angle(pts[11], pts[23], pts[25])
            except: pass
        if pts[12] and pts[24] and pts[26]:
            try: angles['right_hip'] = BiomechanicsV70Classifier.calculate_angle(pts[12], pts[24], pts[26])
            except: pass
        if pts[11] and pts[23] and pts[25]:
            try: angles['left_torso_angle'] = BiomechanicsV70Classifier.calculate_angle(pts[11], pts[23], pts[25])
            except: pass
        if pts[12] and pts[24] and pts[26]:
            try: angles['right_torso_angle'] = BiomechanicsV70Classifier.calculate_angle(pts[12], pts[24], pts[26])
            except: pass
        for key, idx in [('left_shoulder_pos',11),('right_shoulder_pos',12),
                         ('left_hip_pos',23),('right_hip_pos',24),
                         ('left_wrist_pos',15),('right_wrist_pos',16),
                         ('left_heel_pos',29),('right_heel_pos',30)]:
            if pts[idx]: angles[key] = pts[idx]
        if pts[29] and pts[30]: angles['foot_width'] = abs(pts[29][0] - pts[30][0])
        return angles if angles else None

    @staticmethod
    def analyze_joint_dynamics(pose_data_list):
        series = {
            'elbow': [], 'knee': [], 'hip': [],
            'shoulder_y': [], 'hip_y': [], 'torso_angle': [],
            'heel_y': [], 'foot_width': [], 'wrist_y': [],
            'wrist_shoulder_diff': []
        }
        for data in pose_data_list:
            if 'left_elbow' in data: series['elbow'].append(data['left_elbow'])
            elif 'right_elbow' in data: series['elbow'].append(data['right_elbow'])
            if 'left_knee' in data: series['knee'].append(data['left_knee'])
            elif 'right_knee' in data: series['knee'].append(data['right_knee'])
            if 'left_hip' in data: series['hip'].append(data['left_hip'])
            elif 'right_hip' in data: series['hip'].append(data['right_hip'])
            sp = data.get('left_shoulder_pos', data.get('right_shoulder_pos'))
            if sp: series['shoulder_y'].append(sp[1])
            hp = data.get('left_hip_pos', data.get('right_hip_pos'))
            if hp: series['hip_y'].append(hp[1])
            ta = data.get('left_torso_angle', data.get('right_torso_angle'))
            if ta: series['torso_angle'].append(ta)
            hy = data.get('left_heel_pos', data.get('right_heel_pos'))
            if hy: series['heel_y'].append(hy[1])
            if 'foot_width' in data: series['foot_width'].append(data['foot_width'])
            wp = data.get('left_wrist_pos', data.get('right_wrist_pos'))
            if wp:
                series['wrist_y'].append(wp[1])
                if sp: series['wrist_shoulder_diff'].append(sp[1] - wp[1])
        dynamics = {}
        for key, values in series.items():
            if values:
                dynamics[key] = {
                    'mean': float(np.mean(values)), 'range': float(np.max(values) - np.min(values)),
                    'std': float(np.std(values)), 'min': float(np.min(values)), 'max': float(np.max(values))
                }
        return dynamics

    @staticmethod
    def analyze_trajectory_pattern(trajectory, frame_height, frame_width):
        if len(trajectory) < 8: return None
        y_coords = [p[1] for p in trajectory]; x_coords = [p[0] for p in trajectory]
        min_y, max_y = min(y_coords), max(y_coords)
        start_y_pct = (y_coords[0] / frame_height) * 100
        end_y_pct = (y_coords[-1] / frame_height) * 100
        y_range_pct = ((max_y - min_y) / frame_height) * 100
        total_ascent = sum(y_coords[i-1] - y_coords[i] for i in range(1, len(y_coords)) if y_coords[i-1] > y_coords[i])
        total_descent = sum(abs(y_coords[i-1] - y_coords[i]) for i in range(1, len(y_coords)) if y_coords[i-1] < y_coords[i])
        total_movement = total_ascent + total_descent
        return {
            'start_y_pct': start_y_pct, 'end_y_pct': end_y_pct, 'y_range_pct': y_range_pct,
            'starts_low': start_y_pct > 75, 'starts_high': start_y_pct < 35,
            'ends_high': end_y_pct < 45, 'ends_low': end_y_pct > 60,
            'movement_pattern': "oscillating" if total_ascent > 0 and total_descent > 0 else "upward_only",
            'ascent_ratio': total_ascent / total_movement if total_movement > 0 else 0
        }

    @staticmethod
    def classify(trajectory, pose_data_list, frame_height, frame_width,
                 initial_trunk_orientation, initial_barbell_position):
        print("\n" + "="*60)
        print("🎯 v7.0 多信号融合动作分类")
        print("="*60)
        print(f"  躯干: {initial_trunk_orientation}")
        print(f"  杠铃: {initial_barbell_position}")
        jd = BiomechanicsV70Classifier.analyze_joint_dynamics(pose_data_list)
        traj = BiomechanicsV70Classifier.analyze_trajectory_pattern(trajectory, frame_height, frame_width) if trajectory else None
        sq, bp, dl = 0, 0, 0
        if initial_trunk_orientation == 'horizontal': bp += 45
        else: sq += 25; dl += 20
        if initial_barbell_position == 'above_shoulder': sq += 55
        elif initial_barbell_position == 'on_ground': dl += 55
        elif initial_barbell_position == 'above_chest': bp += 55
        elif initial_barbell_position is None:
            if initial_trunk_orientation == 'horizontal': bp += 10
        if traj:
            if traj.get('starts_low') and traj.get('ends_high'): dl += 15; sq += 10
            elif traj.get('starts_high') and traj.get('ends_high'): bp += 10
            yr = traj.get('y_range_pct', 0)
            if yr > 30: sq += 10; dl += 10
            elif yr < 15: bp += 5
        knee_range = jd.get('knee', {}).get('range', 0)
        knee_min = jd.get('knee', {}).get('min', 180)
        elbow_range = jd.get('elbow', {}).get('range', 0)
        torso_angle_range = jd.get('torso_angle', {}).get('range', 0)
        shoulder_y_range = jd.get('shoulder_y', {}).get('range', 0)
        hip_y_range = jd.get('hip_y', {}).get('range', 0)
        wrist_sh_diff_mean = jd.get('wrist_shoulder_diff', {}).get('mean', 0)
        wrist_sh_diff_range = jd.get('wrist_shoulder_diff', {}).get('range', 0)
        print(f"  [信号4] 关节动态: 膝range={knee_range:.1f}° min={knee_min:.1f}° | 肘range={elbow_range:.1f}°")
        if knee_min < 100 and elbow_range < 40: sq += 35
        elif knee_range > 45 and elbow_range < 40: sq += 25
        elif knee_range > 30 and elbow_range < 40: sq += 15
        if shoulder_y_range > frame_height * 0.18: sq += 20; dl += 5
        if elbow_range > 40 and knee_range < 25: bp += 40
        elif elbow_range > 30 and knee_range < 30: bp += 25
        elif elbow_range > 25 and knee_range < 35: bp += 15
        if torso_angle_range > 25 and knee_min > 80 and elbow_range < 35: dl += 35
        elif torso_angle_range > 20 and knee_range > 15 and elbow_range < 40: dl += 25
        if hip_y_range > frame_height * 0.12 and torso_angle_range > 15: dl += 10
        if wrist_sh_diff_mean > frame_height * 0.03:
            if wrist_sh_diff_range < frame_height * 0.15: sq += 15
            else: bp += 10
        elif wrist_sh_diff_mean < -frame_height * 0.05: dl += 15
        if initial_trunk_orientation == 'horizontal' and knee_range > 35: dl = -999
        if initial_trunk_orientation == 'horizontal' and torso_angle_range > 30 and elbow_range < 25: sq = max(sq - 20, 0)
        if initial_trunk_orientation == 'vertical' and elbow_range > 40 and knee_range < 20: dl = max(dl - 20, 0); sq = max(sq - 20, 0)
        print(f"\n  📊 最终评分: 深蹲={sq}, 卧推={bp}, 硬拉={dl}")
        scores = {'Squat': sq, 'Bench Press': bp, 'Deadlift': dl}
        best = max(scores, key=scores.get); best_score = scores[best]
        if best_score < 30:
            if initial_trunk_orientation == 'horizontal': best = "Bench Press"
            elif elbow_range > 35 and knee_range < 25: best = "Bench Press"
            elif knee_min < 100 and elbow_range < 40: best = "Squat"
            elif torso_angle_range > knee_range and elbow_range < 40: best = "Deadlift"
            elif knee_range > elbow_range: best = "Squat"
            else: best = "Deadlift"
        total = sum(max(0, v) for v in scores.values()) + 1e-8
        confidence = (max(0, best_score) / total) * 100
        if best_score < 30: confidence = min(confidence, 50.0)
        print(f"  🎯 判定: {best} (置信度: {confidence:.1f}%)")
        return best, confidence


def analyze_trajectory_features(trajectory, frame_height, frame_width):
    if not trajectory or len(trajectory) < 3:
        return {'stability':'N/A','offset':'N/A','avg_speed':'N/A','max_speed':'N/A',
                'path_smoothness':'N/A','sticking_point':None,'_stability_pct':80.0}
    y_coords = [p[1] for p in trajectory]; x_coords = [p[0] for p in trajectory]
    if len(trajectory) >= 5:
        y_arr = np.array(y_coords, dtype=np.float64); x_arr = np.array(x_coords, dtype=np.float64)
        try:
            coeffs = np.polyfit(y_arr, x_arr, 1)
            x_predicted = np.polyval(coeffs, y_arr)
            residuals = np.abs(x_arr - x_predicted)
            mean_residual = np.mean(residuals)
            max_acceptable = frame_width * 0.08
            stability_pct = max(0, min(100, 100 * (1 - mean_residual / max_acceptable)))
        except: stability_pct = 75.0
    else: stability_pct = 80.0
    stability_str = f"{stability_pct:.1f}%"
    x_range = max(x_coords) - min(x_coords)
    x_offset_pct = (x_range / frame_width) * 100
    if x_offset_pct < 3: offset_str = "居中"
    elif x_offset_pct < 8: offset_str = f"轻微偏移({x_offset_pct:.1f}%)"
    else: offset_str = f"明显偏移({x_offset_pct:.1f}%)"
    speeds = []
    for i in range(1, len(trajectory)):
        dx = trajectory[i][0] - trajectory[i-1][0]; dy = trajectory[i][1] - trajectory[i-1][1]
        speeds.append(math.sqrt(dx**2 + dy**2))
    avg_speed = np.mean(speeds) if speeds else 0
    max_speed = np.max(speeds) if speeds else 0
    if len(speeds) >= 3:
        accels = [abs(speeds[i] - speeds[i-1]) for i in range(1, len(speeds))]
        jerk = np.std(accels)
        smoothness_pct = max(0, min(100, 100 * (1 - jerk / 15.0)))
    else: smoothness_pct = 80.0
    smoothness_str = f"{smoothness_pct:.1f}%"
    sticking_point = None
    if speeds:
        min_idx = np.argmin(speeds)
        if speeds[min_idx] < avg_speed * 0.3 and avg_speed > 2:
            sp_idx = min_idx + 1
            if sp_idx < len(trajectory):
                sticking_point = {
                    'x': int(trajectory[sp_idx][0]), 'y': int(trajectory[sp_idx][1]),
                    'frame_pct': round(sp_idx / len(trajectory) * 100, 1),
                    'description': f"在轨迹 {sp_idx / len(trajectory) * 100:.0f}% 处出现速度骤降"
                }
    return {
        'stability': stability_str, 'offset': offset_str,
        'avg_speed': f"{avg_speed:.1f} px/帧", 'max_speed': f"{max_speed:.1f} px/帧",
        'path_smoothness': smoothness_str, 'sticking_point': sticking_point,
        '_stability_pct': stability_pct, '_avg_speed': avg_speed,
        '_max_speed': max_speed, '_smoothness_pct': smoothness_pct
    }


# ================== FastAPI 应用 ==================
app = FastAPI(title="WL Analysis AI Backend", version="8.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)
UPLOADS_DIR = Path(__file__).parent / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")

# 分块上传会话存储
upload_sessions: Dict[str, dict] = {}


def detect_barbell_position(frame, landmarks_dict):
    if YOLO_AVAILABLE:
        try:
            results = barbell_model(frame, conf=0.3, verbose=False)
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                conf_scores = boxes.conf.cpu().numpy()
                max_idx = np.argmax(conf_scores)
                if conf_scores[max_idx] > 0.3:
                    box = boxes[max_idx]
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    return (int((x1 + x2) / 2), int((y1 + y2) / 2), "yolo")
        except: pass
    return None


def process_video_frame(frame, frame_idx):
    h, w = frame.shape[:2]
    if MEDIAPIPE_AVAILABLE:
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = pose_landmarker.detect(mp_image)
            landmarks_dict = BiomechanicsV70Classifier.extract_landmarks_safe(detection_result)
            pose_data = None
            if landmarks_dict:
                pose_data = BiomechanicsV70Classifier.compute_joint_angles(landmarks_dict, w, h)
                if pose_data: pose_data['frame_idx'] = frame_idx
            barbell_pos = detect_barbell_position(frame, landmarks_dict)
            return barbell_pos, pose_data, landmarks_dict
        except Exception as e:
            if frame_idx % 30 == 0: print(f"⚠️ MediaPipe异常 (帧{frame_idx}): {e}")
    return None, None, None


def gen_thumbnail(video_path, thumb_path):
    try:
        (ffmpeg.input(str(video_path), ss=1)
         .filter('scale', 160, -1)
         .output(str(thumb_path), vframes=1, format='image2', vcodec='mjpeg')
         .overwrite_output()
         .run(capture_stdout=True, capture_stderr=True))
        return thumb_path.exists()
    except: return False


def analyze_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): raise HTTPException(400, "无法打开视频文件")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"📹 视频信息: {fps:.1f}fps, {total_frames}帧, {frame_width}x{frame_height}")
    trajectory = []; pose_data_list = []; landmarks_history = []; all_barbell_positions = []
    MAX_FRAMES = 300
    frames_to_process = min(total_frames, MAX_FRAMES)
    frame_idx = 0
    print("\n📋 步骤1: 逐帧处理 (MediaPipe + YOLO)")
    while cap.isOpened() and frame_idx < frames_to_process:
        ret, frame = cap.read()
        if not ret: break
        if frame_width > 1280:
            scale = 1280 / frame_width
            new_width = 1280; new_height = int(frame_height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
            if frame_idx == 0: frame_height, frame_width = new_height, new_width
        barbell_pos, pose_data, landmarks_dict = process_video_frame(frame, frame_idx)
        if barbell_pos:
            x, y, method = barbell_pos; trajectory.append([x, y])
        if pose_data: pose_data_list.append(pose_data)
        landmarks_history.append(landmarks_dict)
        all_barbell_positions.append(barbell_pos)
        frame_idx += 1
        if frame_idx % 20 == 0: print(f"⏳ 进度: {frame_idx}/{frames_to_process}帧")
    cap.release()
    print(f"✅ 视频处理完成: {frame_idx}帧, 轨迹点: {len(trajectory)}, 姿态帧: {len(pose_data_list)}")
    print("\n📋 步骤2: 多帧躯干朝向判断 (v7.0加权信号)")
    valid_landmarks = [lm for lm in landmarks_history if lm is not None]
    if len(valid_landmarks) >= 5:
        initial_trunk = BiomechanicsV70Classifier.calculate_trunk_orientation_multiframe(valid_landmarks, frame_height, frame_width)
    else:
        first_valid = next((lm for lm in landmarks_history if lm is not None), None)
        initial_trunk = BiomechanicsV70Classifier.calculate_trunk_orientation(first_valid, frame_width, frame_height) if first_valid else 'vertical'
    print("\n📋 步骤3: 多帧杠铃位置投票 (YOLO + 手腕双通道)")
    initial_barbell_pos = BiomechanicsV70Classifier.vote_barbell_position(all_barbell_positions, landmarks_history, frame_width, frame_height)
    print(f"\n🧍 最终躯干朝向: {initial_trunk}")
    print(f"🏋️ 最终杠铃位置: {initial_barbell_pos}")
    return (trajectory, pose_data_list, frame_height, frame_width, initial_trunk, initial_barbell_pos, fps)


def _build_analysis_result(aid, video_path):
    """
    ✅ 核心修复: 统一构建分析结果字典
    确保 /analyze-barbell 和 /merge-and-analyze 返回完全一致的字段
    """
    (trajectory, pose_data_list, frame_height, frame_width,
     initial_trunk, initial_barbell_pos, fps) = analyze_video(video_path)

    if len(pose_data_list) < 10:
        return {
            "success": True, "analysis_id": aid,
            "exercise_type": "Unknown", "exercise_type_zh": "未知动作",
            "confidence": 0, "score": 0,
            "stability": "N/A", "offset": "N/A",
            "avg_speed": "N/A", "max_speed": "N/A",
            "path_smoothness": "N/A", "sticking_point": None, "rpe": 0,
            "feedback": ["⚠️ 姿态数据不足，无法进行有效分析",
                         "💡 请确保: 1. 全身入镜 2. 光线充足 3. 机位稳定"],
            "trajectory": [], "thumbnailUrl": None,
            "videoUrl": f"/uploads/{aid}/{video_path.name}" if video_path.exists() else None,
            "analysis_time": datetime.now().isoformat()
        }

    exercise_type, confidence = BiomechanicsV70Classifier.classify(
        trajectory, pose_data_list, frame_height, frame_width,
        initial_trunk, initial_barbell_pos
    )
    print(f"\n🎯 最终识别结果: {exercise_type} (置信度: {confidence:.1f}%)")

    traj_analysis = analyze_trajectory_features(trajectory, frame_height, frame_width) if trajectory else {
        'stability':'N/A','offset':'N/A','avg_speed':'N/A','max_speed':'N/A',
        'path_smoothness':'N/A','sticking_point':None,'_stability_pct':80.0
    }

    stability_pct = traj_analysis.get('_stability_pct', 80.0)
    score = min(100, max(60, int(stability_pct)))
    rpe = min(10, max(5, 11 - int(stability_pct / 10)))

    feedback = []
    if stability_pct < 70:
        feedback.append(f"⚠️ 轨迹稳定性{stability_pct:.1f}%，建议减重15-25%专注技术")
    elif stability_pct < 80:
        feedback.append(f"🔍 轨迹稳定性{stability_pct:.1f}%，技术有提升空间")
    else:
        feedback.append(f"✅ 轨迹稳定性{stability_pct:.1f}%，技术优秀！")

    if exercise_type == "Bench Press":
        feedback.append("💡 卧推提示: 保持肩胛骨收紧，下放时肘部与躯干呈75°角")
    elif exercise_type == "Squat":
        feedback.append("💡 深蹲提示: 保持核心收紧，膝盖与脚尖方向一致，臀部向后坐")
    elif exercise_type == "Deadlift":
        feedback.append("💡 硬拉提示: 保持背部挺直，杠铃贴近小腿，髋部发力主导")

    sticking = traj_analysis.get('sticking_point')
    if sticking:
        feedback.append(f"🔴 粘滞点: {sticking['description']}")

    smoothness = traj_analysis.get('path_smoothness', 'N/A')
    if smoothness != 'N/A':
        feedback.append(f"📈 路径平滑度: {smoothness}")

    thumb_path = video_path.parent / f"{aid}_thumb.jpg"
    thumb_url = f"/uploads/{aid}/{aid}_thumb.jpg" if gen_thumbnail(video_path, thumb_path) else None
    video_url = f"/uploads/{aid}/{video_path.name}"

    result = {
        "success": True,
        "analysis_id": aid,
        "exercise_type": exercise_type,
        "exercise_type_zh": EXERCISE_ZH_MAP.get(exercise_type, exercise_type),
        "confidence": round(confidence, 1),
        "score": score,
        "stability": traj_analysis['stability'],
        "offset": traj_analysis['offset'],
        "avg_speed": traj_analysis['avg_speed'],
        "max_speed": traj_analysis['max_speed'],
        "path_smoothness": traj_analysis['path_smoothness'],
        "sticking_point": traj_analysis['sticking_point'],
        "rpe": rpe,
        "feedback": feedback,
        "trajectory": trajectory[:100] if trajectory else [],
        "thumbnailUrl": thumb_url,
        "videoUrl": video_url,
        "analysis_time": datetime.now().isoformat()
    }

    # 调试日志
    print(f"🔍 返回字段检查: confidence={result['confidence']}, "
          f"path_smoothness={result['path_smoothness']}, "
          f"feedback类型={type(result['feedback']).__name__}(长度{len(result['feedback'])}), "
          f"trajectory长度={len(result['trajectory'])}")

    return result


# ================== API 路由 ==================

@app.post("/init-upload")
async def init_upload(request: dict):
    file_name = request.get("fileName", f"video-{uuid.uuid4()}.mp4")
    file_size = request.get("fileSize", 0)
    total_chunks = request.get("totalChunks", 1)
    session_id = str(uuid.uuid4())
    session_dir = UPLOADS_DIR / session_id
    session_dir.mkdir(exist_ok=True)
    upload_sessions[session_id] = {
        "fileName": file_name, "fileSize": file_size,
        "totalChunks": total_chunks, "uploadedChunks": [],
        "dir": str(session_dir)
    }
    print(f"📦 初始化上传会话: {session_id}, 文件: {file_name}, 大小: {file_size}, 分块: {total_chunks}")
    return {"success": True, "sessionId": session_id, "fileName": file_name}


@app.get("/get-uploaded-chunks/{session_id}")
async def get_uploaded_chunks(session_id: str):
    session = upload_sessions.get(session_id)
    if not session:
        return {"success": False, "uploadedChunks": []}
    return {"success": True, "uploadedChunks": session["uploadedChunks"]}


@app.post("/upload-chunk")
async def upload_chunk(request: dict):
    session_id = request.get("sessionId")
    chunk_index = request.get("chunkIndex")
    chunk_data = request.get("chunkData")
    if not all([session_id, chunk_index is not None, chunk_data]):
        raise HTTPException(400, "缺少必要参数")
    session = upload_sessions.get(session_id)
    if not session:
        raise HTTPException(404, "会话不存在")
    try:
        chunk_bytes = base64.b64decode(chunk_data)
        chunk_path = Path(session["dir"]) / f"chunk_{chunk_index:04d}"
        with open(chunk_path, "wb") as f:
            f.write(chunk_bytes)
        if chunk_index not in session["uploadedChunks"]:
            session["uploadedChunks"].append(chunk_index)
        return {"success": True, "chunkIndex": chunk_index}
    except Exception as e:
        raise HTTPException(500, f"分块保存失败: {str(e)}")


@app.post("/merge-and-analyze")
async def merge_and_analyze(request: dict):
    """
    ✅ v8.0 核心修复: 合并分块 → 分析 → 返回完整结果
    现在与 /analyze-barbell 返回完全一致的字段结构
    """
    session_id = request.get("sessionId")
    if not session_id:
        raise HTTPException(400, "缺少 sessionId")

    session = upload_sessions.get(session_id)
    if not session:
        raise HTTPException(404, "上传会话不存在或已过期")

    try:
        print(f"\n{'='*60}")
        print(f"🎬 [merge-and-analyze] 开始合并与分析: {session_id}")
        print(f"{'='*60}")

        # 1. 合并分块
        session_dir = Path(session["dir"])
        merged_path = session_dir / session["fileName"]
        total_chunks = session["totalChunks"]

        with open(merged_path, "wb") as outf:
            for i in range(total_chunks):
                chunk_path = session_dir / f"chunk_{i:04d}"
                if not chunk_path.exists():
                    raise HTTPException(400, f"分块 {i} 缺失，上传不完整")
                with open(chunk_path, "rb") as inf:
                    shutil.copyfileobj(inf, outf)

        print(f"✅ 文件合并完成: {merged_path} ({merged_path.stat().st_size} bytes)")

        # 清理分块文件
        for i in range(total_chunks):
            chunk_path = session_dir / f"chunk_{i:04d}"
            if chunk_path.exists():
                chunk_path.unlink()

        # 2. ✅ 调用统一的结果构建函数（与 /analyze-barbell 完全一致）
        result = _build_analysis_result(session_id, merged_path)

        # 清理会话
        if session_id in upload_sessions:
            del upload_sessions[session_id]

        return result

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ merge-and-analyze 失败: {e}")
        import traceback
        traceback.print_exc()
        if session_id in upload_sessions:
            del upload_sessions[session_id]
        raise HTTPException(500, f"合并分析失败: {str(e)}")


@app.post("/analyze-barbell")
async def analyze_barbell(video: UploadFile = File(...)):
    """直接上传分析（与 merge-and-analyze 共用同一结果构建逻辑）"""
    if not MEDIAPIPE_AVAILABLE:
        raise HTTPException(500, "MediaPipe不可用，无法进行分析")

    aid = str(uuid.uuid4())
    tdir = UPLOADS_DIR / aid
    tdir.mkdir(exist_ok=True)

    try:
        print(f"\n{'='*60}")
        print(f"🎬 开始分析: {video.filename}")
        print(f"🆔 分析ID: {aid}")
        print(f"{'='*60}")

        vpath = tdir / video.filename
        with open(vpath, "wb") as f:
            shutil.copyfileobj(video.file, f)
        print(f"💾 视频已保存: {vpath}")

        # ✅ 使用统一的结果构建函数
        return _build_analysis_result(aid, vpath)

    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        shutil.rmtree(tdir, ignore_errors=True)
        raise HTTPException(500, f"分析失败: {str(e)}")


@app.get("/health")
async def health():
    return {
        "status": "OK",
        "timestamp": datetime.now().isoformat(),
        "models": {
            "yolo": "Available" if YOLO_AVAILABLE else "Unavailable",
            "mediapipe": "Available" if MEDIAPIPE_AVAILABLE else "Unavailable"
        },
        "version": "8.0.0 (统一返回格式 + 分块上传)"
    }


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🏋️ WL Analysis AI Backend ")
    print("   1. 新增 _build_analysis_result 统一结果构建")
    print("   2. /merge-and-analyze 返回完整字段(confidence/path_smoothness等)")
    print("   3. /analyze-barbell 与 merge-and-analyze 返回格式完全一致")
    print("   4. 新增 exercise_type_zh 中文字段")
    print("   5. feedback 统一为数组格式")
    print("="*60 + "\n")
    print(f"📊 MediaPipe: {'✅' if MEDIAPIPE_AVAILABLE else '❌'}")
    print(f"📊 YOLOv8: {'✅ (辅助)' if YOLO_AVAILABLE else '❌'}\n")
    uvicorn.run(app, host="0.0.0.0", port=8001)  