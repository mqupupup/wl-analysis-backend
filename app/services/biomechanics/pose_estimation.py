# backend/app/services/biomechanics/pose_estimation.py
"""
姿态估计层：MediaPipe 集成、视频帧处理、关键点提取
（YOLO-pose 作为备用引擎）
"""

import cv2
import numpy as np
import math
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from app.core.config import MAX_FRAMES_TO_PROCESS

# ================================================================
# MediaPipe 初始化（优先使用）
# ================================================================
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    model_path = Path(__file__).parent.parent.parent / "models" / "pose_landmarker.task"
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
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
        min_tracking_confidence=0.3,
    )
    pose_landmarker = vision.PoseLandmarker.create_from_options(options)
    MEDIAPIPE_AVAILABLE = True
    print(f"✅ MediaPipe {mp.__version__} (IMAGE模式) 加载成功")
except Exception as e:
    print(f"❌ MediaPipe 初始化失败: {e}")
    MEDIAPIPE_AVAILABLE = False
    pose_landmarker = None

# ================================================================
# YOLO-pose 初始化（备用引擎）
# ================================================================
YOLO_AVAILABLE = False
yolo_model = None

try:
    from ultralytics import YOLO

    yolo_model_path = Path(__file__).parent.parent.parent / "models" / "yolov8n.pt"
    if not yolo_model_path.exists():
        yolo_model_path.parent.mkdir(parents=True, exist_ok=True)
        print("💡 正在下载 YOLO-pose 模型 (yolov8n.pt)...")
        _tmp_model = YOLO("yolov8n.pt")
        import shutil
        default_path = Path("yolov8n.pt")
        if default_path.exists():
            shutil.move(str(default_path), str(yolo_model_path))
        print(f"✅ YOLO-pose 模型已保存到: {yolo_model_path}")

    yolo_model = YOLO(str(yolo_model_path))
    YOLO_AVAILABLE = True
    print(f"✅ YOLO-pose 模型加载成功: {yolo_model_path.name}")
except Exception as e:
    print(f"⚠️ YOLO-pose 初始化失败（备用引擎，不影响主流程）: {e}")
    YOLO_AVAILABLE = False
    yolo_model = None

# YOLO COCO 17 keypoints → MediaPipe 33 landmarks 索引映射
YOLO_TO_MP_IDX = {
    0: 0, 1: 2, 2: 5, 3: 7, 4: 8,
    5: 11, 6: 12, 7: 13, 8: 14, 9: 15, 10: 16,
    11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28,
}


@dataclass
class FramePoseData:
    """单帧姿态数据"""
    frame_idx: int
    landmarks: Dict[int, Any]
    angles: Dict[str, float]
    positions: Dict[str, List[float]]
    positions_3d: Dict[str, List[float]] = field(default_factory=dict)
    world_landmarks: Dict[str, List[float]] = field(default_factory=dict)
    timestamp_sec: float = 0.0
    # 低阈值 wrist 坐标（visibility>0.15），仅用于 BarPath 检测，不影响次数检测
    wrist_positions_relaxed: Dict[str, List[float]] = field(default_factory=dict)


class PoseEstimator:
    """姿态估计器"""
    
    def __init__(self, engine: str = "mediapipe"):
        """
        Args:
            engine: "mediapipe" (默认优先) | "yolo" | "auto"
        """
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 30.0

        if engine == "auto":
            self.engine = "mediapipe" if MEDIAPIPE_AVAILABLE else "yolo"
        else:
            self.engine = engine

        if self.engine == "mediapipe" and not MEDIAPIPE_AVAILABLE:
            self.engine = "yolo" if YOLO_AVAILABLE else "none"
        if self.engine == "yolo" and not YOLO_AVAILABLE:
            self.engine = "mediapipe" if MEDIAPIPE_AVAILABLE else "none"

        print(f"📌 PoseEstimator 引擎: {self.engine}")

    # ================================================================
    # MediaPipe 方法（原有逻辑，一字未改）
    # ================================================================

    @staticmethod
    def extract_landmarks_safe(detection_result):
        if not detection_result or not detection_result.pose_landmarks:
            return None
        lms = detection_result.pose_landmarks[0]
        landmarks_dict = {}
        for i, l in enumerate(lms):
            if getattr(l, 'visibility', 1.0) > 0.3 and getattr(l, 'presence', 1.0) > 0.3:
                landmarks_dict[i] = {'x': l.x, 'y': l.y, 'z': l.z}
        return landmarks_dict if len(landmarks_dict) >= 10 else None

    @staticmethod
    def extract_world_landmarks(detection_result) -> Dict[str, List[float]]:
        if not detection_result or not hasattr(detection_result, 'pose_world_landmarks'):
            return {}
        world_lms = detection_result.pose_world_landmarks
        if not world_lms or len(world_lms) == 0:
            return {}
        world_lms = world_lms[0]
        result = {}
        key_indices = {
            'left_shoulder': 11, 'right_shoulder': 12, 'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16, 'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26, 'left_ankle': 27, 'right_ankle': 28,
        }
        for name, idx in key_indices.items():
            if idx < len(world_lms):
                lm = world_lms[idx]
                if getattr(lm, 'visibility', 1.0) > 0.3:
                    result[name] = [lm.x, lm.y, lm.z]
        return result

    @staticmethod
    def _extract_wrist_relaxed(detection_result, frame_width, frame_height):
        """低阈值提取 wrist 坐标（visibility>0.15），仅用于 BarPath 检测。

        与 extract_landmarks_safe 不同，这个方法只提取 wrist(15,16)，
        使用更低的 visibility 阈值(0.15)，因为 wrist 经常被杠铃/手背遮挡。
        结果保存到 FramePoseData.wrist_positions_relaxed，BarPath 检测优先使用。
        不影响次数检测（次数检测仍然使用 positions 中的标准阈值 wrist）。
        """
        if not detection_result or not detection_result.pose_landmarks:
            return {}
        lms = detection_result.pose_landmarks[0]
        result = {}
        for idx, name in [(15, 'left_wrist'), (16, 'right_wrist')]:
            if idx < len(lms):
                lm = lms[idx]
                vis = getattr(lm, 'visibility', 1.0)
                pres = getattr(lm, 'presence', 1.0)
                if vis > 0.15 and pres > 0.15:
                    result[name] = [lm.x * frame_width, lm.y * frame_height]
        return result

    # ================================================================
    # YOLO 方法（备用）
    # ================================================================

    def _detect_yolo(self, frame: np.ndarray) -> Optional[Dict[int, Any]]:
        """YOLO-pose 检测，输出统一为 MediaPipe 格式"""
        if not yolo_model:
            return None
        results = yolo_model(frame, verbose=False)
        if not results or len(results) == 0:
            return None
        result = results[0]
        if result.keypoints is None or result.keypoints.data is None:
            return None
        kpts = result.keypoints.data
        if kpts.shape[0] == 0:
            return None

        person_kpts = kpts[0]
        landmarks_dict = {}
        for yolo_idx in range(person_kpts.shape[0]):
            x, y, conf = person_kpts[yolo_idx].cpu().numpy()
            if conf < 0.3:
                continue
            mp_idx = YOLO_TO_MP_IDX.get(yolo_idx)
            if mp_idx is None:
                continue
            landmarks_dict[mp_idx] = {
                'x': float(x) / max(self.frame_width, 1),
                'y': float(y) / max(self.frame_height, 1),
                'z': 0.0,
            }
        return landmarks_dict if len(landmarks_dict) >= 10 else None

    def _extract_world_landmarks_yolo(self, frame: np.ndarray) -> Dict[str, List[float]]:
        """YOLO 近似世界坐标"""
        if not yolo_model:
            return {}
        results = yolo_model(frame, verbose=False)
        if not results or len(results) == 0:
            return {}
        result = results[0]
        if result.keypoints is None or result.keypoints.data is None:
            return {}
        kpts = result.keypoints.data
        if kpts.shape[0] == 0:
            return {}

        person_kpts = kpts[0]
        yolo_name_map = {
            5: 'left_shoulder', 6: 'right_shoulder',
            7: 'left_elbow', 8: 'right_elbow',
            9: 'left_wrist', 10: 'right_wrist',
            11: 'left_hip', 12: 'right_hip',
            13: 'left_knee', 14: 'right_knee',
            15: 'left_ankle', 16: 'right_ankle',
        }
        world_result = {}
        for yolo_idx, name in yolo_name_map.items():
            if yolo_idx >= person_kpts.shape[0]:
                continue
            x, y, conf = person_kpts[yolo_idx].cpu().numpy()
            if conf < 0.3:
                continue
            world_result[name] = [
                float(x) / max(self.frame_width, 1) - 0.5,
                float(y) / max(self.frame_height, 1) - 0.5,
                0.0,
            ]
        return world_result

    # ================================================================
    # 帧数据计算
    # ================================================================

    def compute_frame_data(self, landmarks_dict, frame_idx, timestamp=0.0,
                           world_landmarks_dict=None, wrist_positions_relaxed=None):
        fw, fh = self.frame_width, self.frame_height

        def get_px(idx):
            if idx not in landmarks_dict: return None
            lm = landmarks_dict[idx]
            return [lm['x'] * fw, lm['y'] * fh]

        def get_3d(idx):
            if idx not in landmarks_dict: return None
            lm = landmarks_dict[idx]
            return [lm['x'] * fw, lm['y'] * fh, lm['z'] * fw]

        angles, positions, positions_3d = {}, {}, {}
        key_indices = {
            'left_shoulder': 11, 'right_shoulder': 12, 'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16, 'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26, 'left_ankle': 27, 'right_ankle': 28,
        }
        for name, idx in key_indices.items():
            pt = get_px(idx)
            if pt: positions[name] = pt
            pt3d = get_3d(idx)
            if pt3d: positions_3d[name] = pt3d

 # MediaPipe 的 world_landmarks 是真实的 3D 坐标（米），不受拍摄角度影响
        def get_world(name):
            """从 world_landmarks 获取 3D 坐标"""
            if world_landmarks_dict and name in world_landmarks_dict:
                return world_landmarks_dict[name]  # [x, y, z]
            # 降级：使用 positions_3d
            return positions_3d.get(name)
        
        # 3D 关键点名称到 world_landmarks key 的映射
        idx_to_name = {
            11: 'left_shoulder', 12: 'right_shoulder',
            13: 'left_elbow', 14: 'right_elbow',
            15: 'left_wrist', 16: 'right_wrist',
            23: 'left_hip', 24: 'right_hip',
            25: 'left_knee', 26: 'right_knee',
            27: 'left_ankle', 28: 'right_ankle',
        }
        
        def get_3d_point(idx):
            """获取 3D 点：优先 world_landmarks，降级 positions_3d，再降级 2D"""
            name = idx_to_name.get(idx)
            if name:
                pt = get_world(name)
                if pt:
                    return pt
            # 最终降级：2D
            return get_px(idx)

        # 计算关节角度（3D 优先）
        for name, ai, bi, ci in [
            ('left_elbow', 11, 13, 15), ('right_elbow', 12, 14, 16),
            ('left_knee', 23, 25, 27), ('right_knee', 24, 26, 28),
            ('left_hip', 11, 23, 25), ('right_hip', 12, 24, 26)
        ]:
            pa, pb, pc = get_3d_point(ai), get_3d_point(bi), get_3d_point(ci)
            if pa and pb and pc:
                angles[name] = self.calculate_angle(pa, pb, pc)

        # hip hinge 角
        for prefix, si, hi, ki in [('left', 11, 23, 25), ('right', 12, 24, 26)]:
            s, h, k = get_px(si), get_px(hi), get_px(ki)
            if s and h and k:
                tv = np.array(s) - np.array(h)
                thv = np.array(k) - np.array(h)
                cos = np.dot(tv, thv) / (np.linalg.norm(tv)*np.linalg.norm(thv)+1e-8)
                angles[f'{prefix}_hip_hinge'] = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))

        ls, rs, lh, rh = get_px(11), get_px(12), get_px(23), get_px(24)
        if ls and rs and lh and rh:
            ms = [(ls[0]+rs[0])/2, (ls[1]+rs[1])/2]
            mh = [(lh[0]+rh[0])/2, (lh[1]+rh[1])/2]
            dx, dy = ms[0]-mh[0], mh[1]-ms[1]
            angles['torso_from_vertical'] = math.degrees(math.atan2(abs(dx), max(abs(dy), 1)))

        lw, rw = get_px(15), get_px(16)
        if ls and lw:
            angles['left_shoulder_flexion'] = math.degrees(
                math.atan2(ls[1]-lw[1], max(lw[0]-ls[0], 1)))
        if rs and rw:
            angles['right_shoulder_flexion'] = math.degrees(
                math.atan2(rs[1]-rw[1], max(rw[0]-rs[0], 1)))

        # ================== ✅ 新增：计算肘外展角（上臂与躯干夹角） ==================
        le, re = get_px(13), get_px(14)  # 13=left_elbow, 14=right_elbow
        
        def _calc_upper_arm_torso_angle(shoulder, elbow, hip):
            if not all([shoulder, elbow, hip]):
                return None
            # 躯干向量：髋 -> 肩
            torso_vec = np.array([shoulder[0] - hip[0], shoulder[1] - hip[1]])
            # 上臂向量：肩 -> 肘
            arm_vec = np.array([elbow[0] - shoulder[0], elbow[1] - shoulder[1]])
            
            norm_t = np.linalg.norm(torso_vec)
            norm_a = np.linalg.norm(arm_vec)
            if norm_t == 0 or norm_a == 0:
                return None
                
            cos_angle = np.clip(np.dot(torso_vec, arm_vec) / (norm_t * norm_a), -1.0, 1.0)
            return float(np.degrees(np.arccos(cos_angle)))

        left_tuck = _calc_upper_arm_torso_angle(ls, le, lh)
        right_tuck = _calc_upper_arm_torso_angle(rs, re, rh)
        
        if left_tuck is not None:
            angles['left_upper_arm_torso'] = left_tuck
        if right_tuck is not None:
            angles['right_upper_arm_torso'] = right_tuck
            
        # 计算双侧平均值，供 JSON 规则 ("upper_arm_torso_angle > 75") 直接使用
        tuck_vals = [v for v in [left_tuck, right_tuck] if v is not None]
        if tuck_vals:
            angles['upper_arm_torso_angle'] = sum(tuck_vals) / len(tuck_vals)
        # =======================================================================

        return FramePoseData(
            frame_idx=frame_idx, landmarks=landmarks_dict,
            angles=angles, positions=positions, positions_3d=positions_3d,
            world_landmarks=world_landmarks_dict or {},
            timestamp_sec=timestamp,
            wrist_positions_relaxed=wrist_positions_relaxed or {},
        )
        
    @staticmethod
    def calculate_angle(a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba, bc = a - b, c - b
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))

    # ================================================================
    # 视频处理主流程（MediaPipe 优先，YOLO fallback）
    # ================================================================

    def process_video(self, video_path: Path, max_frames: int = 600) -> List[FramePoseData]:
        """处理视频，提取所有帧的姿态数据"""
        print(f"🎯 分析开始: {video_path}")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError("无法打开视频")

        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"📹 视频信息: {self.frame_width}x{self.frame_height}, {self.fps}fps, {total_frames}帧")
        print(f"🔧 使用引擎: {self.engine}")

        frame_data_list = []
        frame_idx = 0
        mp_hits = 0
        yolo_hits = 0
        # max_frames = min(total_frames, max_frames)
        max_frames = min(MAX_FRAMES_TO_PROCESS, total_frames)

        while cap.isOpened() and frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            lms = None
            world_lms = {}
            wrist_relaxed = {}  # 低阈值 wrist，仅用于 BarPath，不影响次数检测

            # 优先使用 MediaPipe
            if self.engine == "mediapipe" and MEDIAPIPE_AVAILABLE and pose_landmarker:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                res = pose_landmarker.detect(mp_image)
                lms = self.extract_landmarks_safe(res)
                world_lms = self.extract_world_landmarks(res)
                wrist_relaxed = self._extract_wrist_relaxed(res, self.frame_width, self.frame_height)  # 低阈值 wrist
                if lms:
                    mp_hits += 1

            # MediaPipe 未检测到 或 引擎设为 yolo → 使用 YOLO fallback
            if lms is None and self.engine in ("yolo", "mediapipe") and YOLO_AVAILABLE:
                lms = self._detect_yolo(frame)
                if lms:
                    world_lms = self._extract_world_landmarks_yolo(frame)
                    yolo_hits += 1

            if lms:
                frame_data_list.append(
                    self.compute_frame_data(lms, frame_idx, frame_idx / self.fps, world_lms, wrist_relaxed))

            frame_idx += 1

        cap.release()

        print(f"📊 共处理 {frame_idx} 帧，有效姿态帧: {len(frame_data_list)}")
        if mp_hits > 0:
            print(f"   ├─ MediaPipe 命中: {mp_hits} 帧")
        if yolo_hits > 0:
            print(f"   ├─ YOLO fallback 命中: {yolo_hits} 帧")
        return frame_data_list