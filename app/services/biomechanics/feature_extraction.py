# backend/app/services/biomechanics/feature_extraction.py
"""
特征提取层：从姿态数据中提取生物力学特征
"""

import numpy as np
import math
from typing import Dict, List, Optional, Tuple
from scipy.signal import savgol_filter
from .pose_estimation import FramePoseData


class AngularEnergyCalculator:
    """角能量计算器"""
    
    def __init__(self, fps: float = 30.0):
        self.fps = fps

    def compute_angular_energy(self, angle_series: np.ndarray) -> float:
        if len(angle_series) < 3:
            return 0.0
        if len(angle_series) > 7:
            try:
                w = min(7, len(angle_series) if len(angle_series) % 2 != 0 else len(angle_series) - 1)
                smoothed = savgol_filter(angle_series, window_length=w, polyorder=2)
            except Exception:
                smoothed = angle_series
        else:
            smoothed = angle_series
        angular_velocity = np.diff(smoothed) * self.fps
        return float(np.mean(angular_velocity ** 2))

    def compute_shoulder_angular_energy(self, frame_data_list: List[FramePoseData]) -> float:
        series = []
        for fd in frame_data_list:
            for s_name, w_name in [('left_shoulder', 'left_wrist'), ('right_shoulder', 'right_wrist')]:
                s = fd.positions.get(s_name)
                w = fd.positions.get(w_name)
                if s and w:
                    dy = s[1] - w[1]
                    dx = w[0] - s[0]
                    angle = math.degrees(math.atan2(dy, abs(dx) + 1e-8))
                    series.append(angle)
        if len(series) < 3:
            return 0.0
        return self.compute_angular_energy(np.array(series))

    def compute_all(self, frame_data_list: List[FramePoseData]) -> Dict[str, float]:
        angle_names = ['left_elbow', 'right_elbow', 'left_knee', 'right_knee', 'left_hip', 'right_hip']
        energies = {}
        for name in angle_names:
            series = self._get_angle_series(frame_data_list, name)
            energies[name] = self.compute_angular_energy(series)
        energies['shoulder'] = self.compute_shoulder_angular_energy(frame_data_list)
        return energies

    def _get_angle_series(self, frame_data_list: List[FramePoseData], key: str) -> np.ndarray:
        vals = [fd.angles.get(key) for fd in frame_data_list if fd.angles.get(key) is not None]
        return np.array(vals) if vals else np.array([])


class Torso3DAnalyzer:
    """3D 躯干姿态分析器"""
    
    def compute_torso_3d_angles(self, frame_data_list: List[FramePoseData]) -> List[float]:
        angles = []
        for fd in frame_data_list:
            angle = self._compute_single_frame(fd)
            if angle is not None:
                angles.append(angle)
        return angles

    def _compute_single_frame(self, fd: FramePoseData):
        ls = fd.world_landmarks.get('left_shoulder')
        rs = fd.world_landmarks.get('right_shoulder')
        lh = fd.world_landmarks.get('left_hip')
        rh = fd.world_landmarks.get('right_hip')
        if not all([ls, rs, lh, rh]):
            return self._fallback_2d(fd)
        mid_s = [(ls[0]+rs[0])/2, (ls[1]+rs[1])/2, (ls[2]+rs[2])/2]
        mid_h = [(lh[0]+rh[0])/2, (lh[1]+rh[1])/2, (lh[2]+rh[2])/2]
        dx = mid_s[0] - mid_h[0]
        dy = mid_s[1] - mid_h[1]
        dz = mid_s[2] - mid_h[2]
        vertical_component = abs(dy)
        horizontal_component = math.sqrt(dx**2 + dz**2)
        return math.degrees(math.atan2(horizontal_component, vertical_component + 1e-8))

    def _fallback_2d(self, fd: FramePoseData):
        ls = fd.positions.get('left_shoulder')
        rs = fd.positions.get('right_shoulder')
        lh = fd.positions.get('left_hip')
        rh = fd.positions.get('right_hip')
        if not all([ls, rs, lh, rh]):
            return None
        mid_s = [(ls[0]+rs[0])/2, (ls[1]+rs[1])/2]
        mid_h = [(lh[0]+rh[0])/2, (lh[1]+rh[1])/2]
        dx = mid_s[0] - mid_h[0]
        dy = abs(mid_h[1] - mid_s[1])
        return math.degrees(math.atan2(abs(dx), max(dy, 1)))


class V24FeatureExtractor:
    """V24 特征提取器"""
    
    # ✅ 新增：关键点键名别名映射，兼容不同 pose 库的输出格式
    KEY_ALIASES = {
        "left_shoulder":  ["left_shoulder", "LEFT_SHOULDER", "11", "l_shoulder", "shoulder_left"],
        "right_shoulder": ["right_shoulder", "RIGHT_SHOULDER", "12", "r_shoulder", "shoulder_right"],
        "left_elbow":     ["left_elbow", "LEFT_ELBOW", "13", "l_elbow", "elbow_left"],
        "right_elbow":    ["right_elbow", "RIGHT_ELBOW", "14", "r_elbow", "elbow_right"],
        "left_wrist":     ["left_wrist", "LEFT_WRIST", "15", "l_wrist", "wrist_left"],
        "right_wrist":    ["right_wrist", "RIGHT_WRIST", "16", "r_wrist", "wrist_right"],
        "left_hip":       ["left_hip", "LEFT_HIP", "23", "l_hip", "hip_left"],
        "right_hip":      ["right_hip", "RIGHT_HIP", "24", "r_hip", "hip_right"],
        "left_knee":      ["left_knee", "LEFT_KNEE", "25", "l_knee", "knee_left"],
        "right_knee":     ["right_knee", "RIGHT_KNEE", "26", "r_knee", "knee_right"],
        "left_ankle":     ["left_ankle", "LEFT_ANKLE", "27", "l_ankle", "ankle_left"],
        "right_ankle":    ["right_ankle", "RIGHT_ANKLE", "28", "r_ankle", "ankle_right"],
    }

    def __init__(self, fps: float = 30.0, frame_height: int = 1080):
        self.fps = fps
        self.frame_height = frame_height
        self.angular_calc = AngularEnergyCalculator(fps)
        self.torso_3d = Torso3DAnalyzer()

    def extract(self, frame_data_list: List[FramePoseData]) -> Dict[str, float]:
        if len(frame_data_list) < 10:
            return {}
        
        # ✅ 第一步：确保每帧都有完整的基础关节角度（含上肢）
        self._ensure_basic_angles(frame_data_list)
        
        features = {}
        body_height = self._estimate_body_height(frame_data_list)
        features.update(self._compute_angular_energy_features(frame_data_list))
        features.update(self._compute_motion_contribution(features))
        features.update(self._compute_torso_3d_features(frame_data_list))
        features.update(self._compute_rom_features(frame_data_list))
        features.update(self._compute_wrist_trajectory(frame_data_list, body_height))
        features.update(self._compute_body_stability(frame_data_list, body_height))
        features.update(self._compute_descent_onset(frame_data_list))
        features.update(self._compute_shin_angle(frame_data_list))
        features.update(self._compute_posture_features(frame_data_list, body_height))
        features.update(self._compute_hip_hinge_features(frame_data_list))
        features.update(self._compute_wrist_horizontal_features(frame_data_list, body_height))
        return features

    @staticmethod
    def _calc_angle_2d(a, b, c):
        """计算二维平面上 b 点处的夹角 (a-b-c)"""
        if a is None or b is None or c is None:
            return None
        ba = np.array([a[0] - b[0], a[1] - b[1]])
        bc = np.array([c[0] - b[0], c[1] - b[1]])
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        if norm_ba < 1e-8 or norm_bc < 1e-8:
            return None
        cos_angle = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))

    def _resolve_position(self, fd: FramePoseData, canonical_key: str):
        """按优先级尝试多个别名，返回第一个找到的坐标"""
        aliases = self.KEY_ALIASES.get(canonical_key, [canonical_key])
        for alias in aliases:
            val = fd.positions.get(alias)
            if val is not None:
                return val
        return None

    def _ensure_basic_angles(self, frame_data_list: List[FramePoseData]):
        """
        确保每帧 FramePoseData.angles 中包含完整的基础关节角度。
        如果 pose_estimation 层已经计算过则跳过，否则在此补全。
        """
        # 🔍 诊断日志：打印第一帧的实际可用键名，帮助排查 pose 库输出格式
        if frame_data_list:
            fd0 = frame_data_list[0]
            pos_keys = sorted(fd0.positions.keys()) if hasattr(fd0, 'positions') and fd0.positions else []
            print(f"🔑 [诊断] 第一帧 positions 可用键 ({len(pos_keys)}): {pos_keys}")
            angle_keys = sorted(fd0.angles.keys()) if hasattr(fd0, 'angles') and fd0.angles else []
            print(f"🔑 [诊断] 第一帧 angles 已有键 ({len(angle_keys)}): {angle_keys}")

        ANGLE_DEFINITIONS = [
            # (输出键名, 点A, 顶点B, 点C)
            ("left_elbow", "left_shoulder", "left_elbow", "left_wrist"),
            ("right_elbow", "right_shoulder", "right_elbow", "right_wrist"),
            ("left_shoulder", "left_elbow", "left_shoulder", "left_hip"),
            ("right_shoulder", "right_elbow", "right_shoulder", "right_hip"),
            ("left_knee", "left_hip", "left_knee", "left_ankle"),
            ("right_knee", "right_hip", "right_knee", "right_ankle"),
            ("left_hip", "left_shoulder", "left_hip", "left_knee"),
            ("right_hip", "right_shoulder", "right_hip", "right_knee"),
        ]
        
        for fd in frame_data_list:
            if not hasattr(fd, 'angles') or fd.angles is None:
                fd.angles = {}
            
            for key, pa, pb, pc in ANGLE_DEFINITIONS:
                # 仅在键不存在或为 None 时补算
                if key not in fd.angles or fd.angles[key] is None:
                    # ✅ 使用别名解析获取关键点坐标
                    a = self._resolve_position(fd, pa)
                    b = self._resolve_position(fd, pb)
                    c = self._resolve_position(fd, pc)
                    val = self._calc_angle_2d(a, b, c)
                    if val is not None:
                        fd.angles[key] = val
            
            # 计算双侧平均角度（供 Rep 检测使用）
            le = fd.angles.get("left_elbow")
            re = fd.angles.get("right_elbow")
            if le is not None and re is not None and "elbow_angle_avg" not in fd.angles:
                fd.angles["elbow_angle_avg"] = (le + re) / 2.0
            
            ls_val = fd.angles.get("left_shoulder")
            rs_val = fd.angles.get("right_shoulder")
            if ls_val is not None and rs_val is not None and "shoulder_flexion_avg" not in fd.angles:
                fd.angles["shoulder_flexion_avg"] = (ls_val + rs_val) / 2.0
            
            lk = fd.angles.get("left_knee")
            rk = fd.angles.get("right_knee")
            if lk is not None and rk is not None and "knee_angle_avg" not in fd.angles:
                fd.angles["knee_angle_avg"] = (lk + rk) / 2.0
            
            lh_val = fd.angles.get("left_hip")
            rh_val = fd.angles.get("right_hip")
            if lh_val is not None and rh_val is not None and "hip_angle_avg" not in fd.angles:
                fd.angles["hip_angle_avg"] = (lh_val + rh_val) / 2.0

            # 计算上臂-躯干夹角（elbow tuck / shoulder abduction）
            for side in ["left", "right"]:
                key = f"{side}_upper_arm_torso"
                if key not in fd.angles or fd.angles[key] is None:
                    val = self._calculate_elbow_tuck_angle(fd.positions, side=side)
                    if val is not None:
                        fd.angles[key] = val

    def _compute_angular_energy_features(self, frame_data_list):
        energies = self.angular_calc.compute_all(frame_data_list)
        elbow_e = (energies.get('left_elbow', 0) + energies.get('right_elbow', 0)) / 2
        knee_e = (energies.get('left_knee', 0) + energies.get('right_knee', 0)) / 2
        hip_e = (energies.get('left_hip', 0) + energies.get('right_hip', 0)) / 2
        shoulder_e = energies.get('shoulder', 0)
        upper = elbow_e + shoulder_e
        lower = knee_e + hip_e
        total = upper + lower + 1e-8
        return {
            'upper_angular_energy_ratio': round(upper / total, 4),
            'elbow_angular_energy': round(elbow_e, 2),
            'shoulder_angular_energy': round(shoulder_e, 2),
            'knee_angular_energy': round(knee_e, 2),
            'hip_angular_energy': round(hip_e, 2),
            'elbow_to_hip_angular_ratio': round(elbow_e / (hip_e + 1e-8), 3),
            'knee_to_hip_angular_ratio': round(knee_e / (hip_e + 1e-8), 3),
        }

    def _compute_motion_contribution(self, features):
        elbow = features.get('elbow_angular_energy', 0)
        knee = features.get('knee_angular_energy', 0)
        hip = features.get('hip_angular_energy', 0)
        total = elbow + knee + hip + 1e-8
        return {
            'upper_motion_ratio': round(elbow / total, 4),
            'knee_motion_ratio': round(knee / total, 4),
            'hip_motion_ratio': round(hip / total, 4),
            'lower_motion_ratio': round((knee + hip) / total, 4),
        }

    def _compute_torso_3d_features(self, frame_data_list):
        angles = self.torso_3d.compute_torso_3d_angles(frame_data_list)
        if not angles:
            return {'torso_pitch_3d_mean': 45.0, 'torso_pitch_3d_rom': 0.0,
                    'torso_pitch_3d_max': 45.0, 'torso_pitch_3d_min': 45.0}
        arr = np.array(angles)
        return {
            'torso_pitch_3d_mean': round(float(np.median(arr)), 1),
            'torso_pitch_3d_rom': round(float(np.ptp(arr)), 1),
            'torso_pitch_3d_max': round(float(np.percentile(arr, 95)), 1),
            'torso_pitch_3d_min': round(float(np.percentile(arr, 5)), 1),
        }

    def _compute_rom_features(self, frame_data_list):
        rom_pairs = {'knee': ('left_knee', 'right_knee'), 'hip': ('left_hip', 'right_hip'),
                     'elbow': ('left_elbow', 'right_elbow')}
        roms = {}
        for joint, (lk, rk) in rom_pairs.items():
            lv = self._get_angle_series(frame_data_list, lk)
            rv = self._get_angle_series(frame_data_list, rk)
            lr = float(np.ptp(lv)) if len(lv) > 2 else 0.0
            rr = float(np.ptp(rv)) if len(rv) > 2 else 0.0
            roms[joint] = (lr + rr) / 2
        kr, hr, er = roms.get('knee', 0), roms.get('hip', 0), roms.get('elbow', 0)
        return {
            'knee_rom_bilateral': round(kr, 1), 'hip_rom_bilateral': round(hr, 1),
            'elbow_rom_bilateral': round(er, 1),
            'hip_knee_rom_ratio': round(hr / (kr + 1e-8), 3),
            'elbow_hip_rom_ratio': round(er / (hr + 1e-8), 3),
            'upper_lower_rom_ratio': round(er / (kr + hr + 1e-8), 3),
        }

    def _compute_wrist_trajectory(self, frame_data_list, body_height):
        bh = body_height + 1e-8
        wx, wy = [], []
        for fd in frame_data_list:
            lw, rw = fd.positions.get('left_wrist'), fd.positions.get('right_wrist')
            if lw and rw:
                wx.append((lw[0]+rw[0])/2/bh); wy.append((lw[1]+rw[1])/2/bh)
            elif lw:
                wx.append(lw[0]/bh); wy.append(lw[1]/bh)
            elif rw:
                wx.append(rw[0]/bh); wy.append(rw[1]/bh)
        if len(wx) < 5:
            return {'wrist_horizontal_variance': 0.0, 'wrist_vertical_variance': 0.0,
                    'wrist_vertical_dominance': 0.5, 'wrist_y_range_norm': 0.0,
                    'wrist_above_shoulder_pct': 0.0}
        xa, ya = np.array(wx), np.array(wy)
        xv, yv = float(np.var(xa)), float(np.var(ya))
        total = xv + yv + 1e-8

        above_count = 0
        total_count = 0
        for fd in frame_data_list:
            for sn, wn in [('left_shoulder', 'left_wrist'), ('right_shoulder', 'right_wrist')]:
                s, w = fd.positions.get(sn), fd.positions.get(wn)
                if s and w:
                    total_count += 1
                    if w[1] < s[1]:
                        above_count += 1
        above_pct = above_count / (total_count + 1e-8)

        return {
            'wrist_horizontal_variance': round(xv, 6),
            'wrist_vertical_variance': round(yv, 6),
            'wrist_vertical_dominance': round(yv / total, 4),
            'wrist_y_range_norm': round(float(np.ptp(ya)), 4),
            'wrist_above_shoulder_pct': round(float(above_pct), 3),
        }
    
    def _calculate_elbow_tuck_angle(self, landmarks, side="left"):
        """
        计算肘外展角（上臂与躯干的夹角）
        - 接近 0°: 手臂完全夹紧身体
        - 接近 90°: 手臂完全平展（T型）
        """
        shoulder = landmarks.get(f"{side}_shoulder")
        elbow = landmarks.get(f"{side}_elbow")
        hip = landmarks.get(f"{side}_hip")
        
        if not all([shoulder, elbow, hip]):
            return None
            
        torso_vec = np.array([shoulder[0] - hip[0], shoulder[1] - hip[1]])
        arm_vec = np.array([elbow[0] - shoulder[0], elbow[1] - shoulder[1]])
        
        dot_product = np.dot(torso_vec, arm_vec)
        norm_torso = np.linalg.norm(torso_vec)
        norm_arm = np.linalg.norm(arm_vec)
        
        if norm_torso == 0 or norm_arm == 0:
            return None
            
        cos_angle = np.clip(dot_product / (norm_torso * norm_arm), -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg

    def _compute_body_stability(self, frame_data_list, body_height):
        bh = body_height + 1e-8
        hy, sy = [], []
        for fd in frame_data_list:
            lh, rh = fd.positions.get('left_hip'), fd.positions.get('right_hip')
            if lh and rh: hy.append((lh[1]+rh[1])/2/bh)
            ls, rs = fd.positions.get('left_shoulder'), fd.positions.get('right_shoulder')
            if ls and rs: sy.append((ls[1]+rs[1])/2/bh)
        hr = float(np.ptp(hy)) if hy else 0.0
        sr = float(np.ptp(sy)) if sy else 0.0
        return {
            'body_vertical_motion': round(max(hr, sr), 4),
            'hip_vertical_rom_norm': round(hr, 4),
            'shoulder_vertical_rom_norm': round(sr, 4),
        }

    def _compute_descent_onset(self, frame_data_list):
        ks = self._get_angle_series(frame_data_list, 'left_knee')
        hs = self._get_angle_series(frame_data_list, 'left_hip')
        if len(ks) < 20 or len(hs) < 20:
            return {'descent_knee_leads_ratio': 0.5}
        ml = min(len(ks), len(hs))
        ks, hs = ks[:ml], hs[:ml]
        if ml > 7:
            try:
                w = min(11, ml if ml % 2 != 0 else ml - 1)
                ks = savgol_filter(ks, window_length=w, polyorder=2)
                hs = savgol_filter(hs, window_length=w, polyorder=2)
            except Exception:
                pass
        kv = np.diff(ks) * self.fps
        hv = np.diff(hs) * self.fps
        ko = self._find_sustained_onsets(kv, -10.0, 6)
        ho = self._find_sustained_onsets(hv, -10.0, 6)
        if not ko or not ho:
            return {'descent_knee_leads_ratio': 0.5}
        kl, t = 0, 0
        for k in ko:
            ch = min(ho, key=lambda h: abs(h - k))
            lag = ch - k
            t += 1
            if lag > 2: kl += 1
            elif lag < -2: pass
            else: kl += 0.5
        return {'descent_knee_leads_ratio': round(kl / (t + 1e-8), 3)}

    def _find_sustained_onsets(self, velocity, threshold, min_consecutive):
        onsets, cc, oc, ia = [], 0, None, False
        for i, v in enumerate(velocity):
            if v < threshold:
                if cc == 0: oc = i
                cc += 1
                if cc >= min_consecutive and not ia:
                    onsets.append(oc); ia = True
            else:
                cc = 0; oc = None
                if v > threshold * 0.5: ia = False
        return onsets

    def _compute_shin_angle(self, frame_data_list):
        sa = []
        for fd in frame_data_list:
            for kn, an in [('left_knee', 'left_ankle'), ('right_knee', 'right_ankle')]:
                k, a = fd.positions.get(kn), fd.positions.get(an)
                if k and a:
                    dx, dy = a[0]-k[0], a[1]-k[1]
                    sa.append(math.degrees(math.atan2(abs(dx), abs(dy)+1e-8)))
        if not sa:
            return {'shin_angle_max': 0.0, 'shin_angle_rom': 0.0}
        arr = np.array(sa)
        return {'shin_angle_max': round(float(np.percentile(arr, 95)), 1),
                'shin_angle_rom': round(float(np.ptp(arr)), 1)}

    def _compute_posture_features(self, frame_data_list, body_height):
        torso_angles, sh_diffs, thr = [], [], []
        for fd in frame_data_list:
            ls, rs = fd.positions.get('left_shoulder'), fd.positions.get('right_shoulder')
            lh, rh = fd.positions.get('left_hip'), fd.positions.get('right_hip')
            if ls and rs and lh and rh:
                ms = [(ls[0]+rs[0])/2, (ls[1]+rs[1])/2]
                mh = [(lh[0]+rh[0])/2, (lh[1]+rh[1])/2]
                dx, dy = ms[0]-mh[0], abs(mh[1]-ms[1])
                torso_angles.append(math.degrees(math.atan2(dy, abs(dx)+1e-8)))
                sh_diffs.append(dy / (self.frame_height+1e-8))
                thr.append(abs(dx) / (dy+1e-8))
        if not torso_angles:
            return {'posture_lying_score': 0.5, 'torso_horizontal_ratio': 0.5}
        tm = float(np.median(torso_angles))
        sm = float(np.median(sh_diffs))
        hm = float(np.median(thr))
        as_ = 1.0 - np.clip(tm / 90.0, 0, 1)
        ds = 1.0 - np.clip(sm / 0.4, 0, 1)
        ls_ = as_ * 0.6 + ds * 0.4
        return {'posture_lying_score': round(float(ls_), 4),
                'torso_horizontal_ratio': round(hm, 4)}

    def _compute_hip_hinge_features(self, frame_data_list):
        ha = []
        for fd in frame_data_list:
            for side in [('left_shoulder', 'left_hip', 'left_knee'),
                         ('right_shoulder', 'right_hip', 'right_knee')]:
                s, h, k = fd.positions.get(side[0]), fd.positions.get(side[1]), fd.positions.get(side[2])
                if s and h and k:
                    tv = np.array(s) - np.array(h)
                    thv = np.array(k) - np.array(h)
                    cos = np.dot(tv, thv) / (np.linalg.norm(tv)*np.linalg.norm(thv)+1e-8)
                    ha.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
        if not ha:
            return {'hip_hinge_mean': 45.0, 'hip_hinge_rom': 0.0, 'hip_hinge_max': 45.0}
        arr = np.array(ha)
        return {'hip_hinge_mean': round(float(np.median(arr)), 1),
                'hip_hinge_rom': round(float(np.ptp(arr)), 1),
                'hip_hinge_max': round(float(np.percentile(arr, 95)), 1)}

    def _compute_wrist_horizontal_features(self, frame_data_list, body_height):
        bh = body_height + 1e-8
        wx = []
        for fd in frame_data_list:
            lw, rw = fd.positions.get('left_wrist'), fd.positions.get('right_wrist')
            if lw and rw: wx.append((lw[0]+rw[0])/2/bh)
            elif lw: wx.append(lw[0]/bh)
            elif rw: wx.append(rw[0]/bh)
        if len(wx) < 5:
            return {'wrist_horizontal_rom_norm': 0.0}
        return {'wrist_horizontal_rom_norm': round(float(np.ptp(np.array(wx))), 4)}

    def _get_angle_series(self, frame_data_list, key):
        vals = [fd.angles.get(key) for fd in frame_data_list if fd.angles.get(key) is not None]
        return np.array(vals) if vals else np.array([])

    def _estimate_body_height(self, frame_data_list):
        hs = []
        for fd in frame_data_list:
            ls, la = fd.positions.get('left_shoulder'), fd.positions.get('left_ankle')
            if ls and la:
                hs.append(math.sqrt((ls[0]-la[0])**2 + (ls[1]-la[1])**2))
        return float(np.median(hs)) if hs else float(self.frame_height * 0.6)