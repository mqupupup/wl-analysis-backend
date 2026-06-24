from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict
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

# ================== 1. 鲁棒动作分类器  ==================
class RobustExerciseClassifier:
    """
    放宽预过滤对卧推的误杀 + 增强卧推姿态缺失时的兜底
    """
    
    @staticmethod
    def calculate_angle(a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba, bc = a - b, c - b
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    
    @staticmethod
    def _is_non_exercise_video(trajectory, pose_data, frame_height):
        """
        : 仅保留高置信度负向过滤
        移除：关键点覆盖率检查（卧推肘部易丢失）
        放宽：高频振荡阈值（卧推本身是周期性运动）
        """
        if not trajectory or len(trajectory) < 15:
            return True, "帧数不足"
            
        y_coords = [p[1] for p in trajectory]
        x_coords = [p[0] for p in trajectory]
        
        # 1. 运动幅度过小 → 保留，但降低门槛
        y_range_px = max(y_coords) - min(y_coords)
        if y_range_px < 20:  #  从30降至20，避免小幅度卧推被误杀
            return True, "运动幅度不足"
        
        # 2. 水平抖动过大 → 保留
        x_std_px = np.std(x_coords)
        if x_std_px > frame_height * 0.30:  #  从0.25放宽到0.30
            return True, "镜头不稳定"
        
        # 3. 高频振荡 → 大幅放宽阈值
        if len(y_coords) > 30:
            diffs = np.diff(y_coords)
            sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
            freq_ratio = sign_changes / len(y_coords)
            if freq_ratio > 0.6:
                return True, "高频非力量动作"
        # 卧推时肘部被杠铃遮挡是常态，不应作为否决条件
        
        return False, ""

    @staticmethod
    def classify(trajectory, pose_data, frame_height):
        # Step 0: 预过滤
        is_non_ex, reason = RobustExerciseClassifier._is_non_exercise_video(trajectory, pose_data, frame_height)
        if is_non_ex:
            print(f"🚫 预过滤拦截: {reason}")
            return "Unknown"
        
        # --- 轨迹特征 ---
        y_coords = [p[1] for p in trajectory]
        x_coords = [p[0] for p in trajectory]
        y_range = max(y_coords) - min(y_coords)
        x_std = np.std(x_coords)
        rel_y_range = (y_range / frame_height) * 100
        rel_x_std = (x_std / frame_height) * 100
        start_y_ratio = (y_coords[0] / frame_height) * 100
        
        # --- 姿态特征统计 ---
        elbow_angles = [d['elbow_angle'] for d in pose_data if 'elbow_angle' in d]
        hip_angles = [d['hip_angle'] for d in pose_data if 'hip_angle' in d]
        knee_angles = [d['knee_angle'] for d in pose_data if 'knee_angle' in d]
        torso_angles = [d['torso_angle'] for d in pose_data if 'torso_angle' in d]
        
        has_elbow = len(elbow_angles) >= 3   #  从5降至3，适应卧推肘部丢失
        has_hip = len(hip_angles) >= 5
        has_knee = len(knee_angles) >= 5
        has_torso = len(torso_angles) >= 3   #  从5降至3
        
        min_elbow = np.min(elbow_angles) if has_elbow else 180
        avg_elbow = np.mean(elbow_angles) if has_elbow else 180
        max_elbow = np.max(elbow_angles) if has_elbow else 0
        elbow_range = max_elbow - min_elbow if has_elbow else 0
        
        hip_range = (max(hip_angles) - min(hip_angles)) if has_hip else 0
        min_knee = np.min(knee_angles) if has_knee else 180
        knee_range = (max(knee_angles) - min(knee_angles)) if has_knee else 0
        avg_torso = np.mean(torso_angles) if has_torso else 45
        
        # ========== 多维加权评分系统  ==========
        scores = {"Bench Press": 0, "Squat": 0, "Deadlift": 0}
        
        # 【卧推】: 增加"仅有轨迹+少量肘部"的中间态兜底
        bench_score = 0
        confidence_decay = 1.0
        
        if has_elbow and has_torso:
            # ✅ 完整姿态
            confidence_decay = 1.0
            if avg_torso > 50: bench_score += 35
            if min_elbow < 110 and elbow_range > 15: bench_score += 30
            if 8 < rel_y_range < 45: bench_score += 20
            if rel_x_std < 15: bench_score += 15
            
        elif has_elbow and not has_torso:
            # ⚠️ 有肘无躯干（卧推常见：侧面拍摄躯干被遮挡）
            confidence_decay = 0.75  #  从0.7微升
            if min_elbow < 110 and elbow_range > 15: bench_score += 40  #  放宽肘角上限
            if 8 < rel_y_range < 45 and rel_x_std < 15: bench_score += 35  #  放宽范围
            if start_y_ratio < 75: bench_score += 25  #  从70放宽到75
            
        elif not has_elbow and has_torso:
            #  新增：有躯干无肘部（正面拍摄肘部被杠铃遮挡）
            confidence_decay = 0.6
            if avg_torso > 50: bench_score += 30
            if 8 < rel_y_range < 45 and rel_x_std < 12: bench_score += 30
            if start_y_ratio < 70: bench_score += 20
            
        else:
            #  纯轨迹兜底：强约束 + 大幅衰减
            confidence_decay = 0.4
            if (10 < rel_y_range < 35 and 
                rel_x_std < 8 and 
                start_y_ratio < 60 and 
                y_range > frame_height * 0.08):
                bench_score += 50
            else:
                bench_score = 0
                
        scores["Bench Press"] = int(bench_score * confidence_decay)
        
        # 【深蹲】保持不变
        squat_score = 0
        if has_knee and min_knee < 100 and knee_range > 40:
            squat_score += 40
        if has_torso and avg_torso < 65:
            squat_score += 25
        if has_hip and hip_range > 30:
            squat_score += 20
        if rel_y_range > 20:
            squat_score += 15
        scores["Squat"] = squat_score
        
        # 【硬拉】保持不变
        dead_score = 0
        if start_y_ratio > 55 and rel_y_range > 20:
            dead_score += 30
        if has_elbow and avg_elbow > 140 and elbow_range < 20:
            dead_score += 30
        if rel_x_std < 6:
            dead_score += 25
        if has_torso and 30 < avg_torso < 70:
            dead_score += 15
        scores["Deadlift"] = dead_score
        
        best = max(scores, key=scores.get)
        best_score = scores[best]
        
        print(f"📊 分类得分: {scores}")
        print(f"🦴 姿态: 肘{avg_elbow:.0f}°(幅{elbow_range:.0f}°) 膝{min_knee:.0f}° 躯干{avg_torso:.0f}° 髋幅{hip_range:.0f}°")
        print(f"📏 轨迹: Y范围{rel_y_range:.1f}% X稳定{rel_x_std:.1f}% 起始{start_y_ratio:.1f}%")
        print(f"📈 数据量: 肘{len(elbow_angles)}帧 髋{len(hip_angles)}帧 膝{len(knee_angles)}帧 躯干{len(torso_angles)}帧")
        
        threshold = 35 if best == "Bench Press" else 45
        return best if best_score >= threshold else "Unknown"


# ================== 2. FastAPI 应用初始化 ==================
app = FastAPI(title="WL Analysis AI Backend", version="2.3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

UPLOADS_DIR = Path(__file__).parent / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")

# MediaPipe Tasks API 初始化
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
        base_options=base_options, running_mode=vision.RunningMode.VIDEO,
        num_poses=1, min_pose_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )
    pose_landmarker = vision.PoseLandmarker.create_from_options(options)
    MEDIAPIPE_AVAILABLE = True
    print(f"✅ MediaPipe {mp.__version__} (Tasks API) 加载成功")
except Exception as e:
    print(f"❌ MediaPipe 初始化失败: {e}")
    MEDIAPIPE_AVAILABLE = False
    pose_landmarker = None

# YOLOv8 初始化
try:
    from ultralytics import YOLO
    yolo_path = Path(__file__).parent / "yolov8n.pt"
    if yolo_path.exists():
        barbell_model = YOLO(str(yolo_path))
        YOLO_AVAILABLE = True
        print("✅ YOLOv8 加载成功")
    else:
        YOLO_AVAILABLE = False
        print("⚠️ YOLOv8 模型不存在，使用手腕估算")
except Exception as e:
    YOLO_AVAILABLE = False
    print(f"⚠️ YOLOv8 加载失败: {e}")


# ================== 3. 数据模型与工具函数 ==================
class AnalysisResult(BaseModel):
    success: bool; analysis_id: str; exercise_type: str; score: int
    stability: str; offset: str; avg_speed: str; max_speed: str
    sticking_point: Optional[Dict]; rpe: int; feedback: List[str]
    trajectory: List[List[float]]; thumbnailUrl: Optional[str]
    videoUrl: Optional[str]; analysis_time: str

UNKNOWN_RESPONSE = {
    "success": True,
    "exercise_type": "Unknown",
    "score": 0,
    "stability": "N/A", "offset": "N/A", 
    "avg_speed": "N/A", "max_speed": "N/A",
    "sticking_point": None, "rpe": 0,
    "feedback": [
        "⚠️ 未识别到卧推/深蹲/硬拉动作",
        "💡 请上传清晰的三大项训练视频",
        "🔍 确保全身入镜、光线充足、机位固定"
    ],
    "trajectory": []
}

def detect_barbell_position(frame, landmarks_dict):
    if YOLO_AVAILABLE:
        try:
            results = barbell_model(frame, verbose=False)
            for r in results:
                for box in r.boxes:
                    x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
                    return (int((x1+x2)/2), int((y1+y2)/2), "yolo")
        except: pass
    
    if landmarks_dict:
        try:
            h, w, _ = frame.shape
            lw = landmarks_dict[15]; rw = landmarks_dict[16]
            bx = int((lw.x*w + rw.x*w) / 2)
            by = int((lw.y*h + rw.y*h) / 2)
            return (bx, by, "estimated")
        except: pass
    return None

def extract_landmarks(detection_result):
    if not detection_result or not detection_result.pose_landmarks:
        return None
    lms = detection_result.pose_landmarks[0]
    return {i: type('L',(),{'x':l.x,'y':l.y,'z':l.z,'visibility':getattr(l,'visibility',1.0)})() for i,l in enumerate(lms)}

def compute_pose_metrics(lms, w, h):
    def pt(idx): 
        l = lms[idx]; return [l.x*w, l.y*h]
    
    calc = RobustExerciseClassifier.calculate_angle
    result = {}
    
    try:
        shoulder, elbow, wrist = pt(11), pt(13), pt(15)
        result['elbow_angle'] = calc(shoulder, elbow, wrist)
    except: pass
    
    try:
        shoulder, hip, knee = pt(11), pt(23), pt(25)
        result['hip_angle'] = calc(shoulder, hip, knee)
    except: pass
    
    try:
        hip, knee, ankle = pt(23), pt(25), pt(27)
        result['knee_angle'] = calc(hip, knee, ankle)
    except: pass
    
    try:
        shoulder, hip = pt(11), pt(23)
        torso_vec = np.array([shoulder[0]-hip[0], shoulder[1]-hip[1]])
        vertical = np.array([0, -1])
        cos_t = np.dot(torso_vec, vertical) / (np.linalg.norm(torso_vec) + 1e-8)
        result['torso_angle'] = abs(np.degrees(np.arccos(np.clip(cos_t, -1, 1))))
    except: pass
    
    return result

def is_valid_video(trajectory, pose_data):
    if not trajectory or len(trajectory) < 15: return False
    ys = [p[1] for p in trajectory]
    if max(ys)-min(ys) < 40: return False
    if np.std([p[0] for p in trajectory]) > 250: return False
    return True

def analyze_trajectory(trajectory):
    if not trajectory or len(trajectory) < 5:
        return {'stability':'N/A','offset':'N/A','avg_speed':'N/A','max_speed':'N/A','sticking_point':None}
    xs = [p[0] for p in trajectory]
    x_std = np.std(xs)
    stab = max(0, 100-(x_std/10)*2)
    x_mean, x_med = np.mean(xs), np.median(xs)
    off_dir = "左偏" if x_mean < x_med else "右偏"
    
    speeds = []
    for i in range(1,len(trajectory)):
        d = math.sqrt((trajectory[i][0]-trajectory[i-1][0])**2 + (trajectory[i][1]-trajectory[i-1][1])**2)
        speeds.append(d*30)
    avg_s, max_s = (np.mean(speeds), max(speeds)) if speeds else (0,0)
    
    sp = None
    if len(speeds)>10:
        mi = np.argmin(speeds[5:-5])+5
        if speeds[mi] < np.mean(speeds)*0.3:
            sp = {'frame':int(mi),'position':f"{(mi/len(speeds))*100:.1f}%"}
    
    return {'stability':f"{stab:.1f}%",'offset':f"{off_dir} {abs(x_mean-x_med):.1f}px",
            'avg_speed':f"{avg_s:.2f}px/s",'max_speed':f"{max_s:.2f}px/s",'sticking_point':sp}

def calc_rpe(traj_data, pose_data):
    try: stab = float(traj_data['stability'].replace('%',''))
    except: stab = 80.0
    if pose_data:
        es = 100 - np.std([d['elbow_angle'] for d in pose_data if 'elbow_angle' in d])
        rpe = 8 + (100-stab)/15 + (100-es)/20
    else:
        rpe = 8 + (100-stab)/10
    return min(10, max(5, int(rpe)))

def gen_feedback(traj, pose, ex_type):
    fb = []
    if ex_type == "Unknown":
        return UNKNOWN_RESPONSE["feedback"]
    stab = float(traj['stability'].replace('%',''))
    if stab < 85: fb.append(f"⚠️ 轨迹稳定性{stab:.1f}%，建议减重或改进技术")
    if traj['sticking_point']: fb.append(f"⚠️ 卡点位于{traj['sticking_point']['position']}")
    if pose and ex_type=="Bench Press":
        valid = [d for d in pose if 'elbow_angle' in d and 50<=d['elbow_angle']<=110]
        if valid:
            ae = np.mean([d['elbow_angle'] for d in valid])
            if ae<60: fb.append(f"⚠️ 肘角{ae:.0f}°过小，肩部压力大")
            elif ae>90: fb.append(f"⚠️ 肘角{ae:.0f}°过大，建议70-80°")
    if not fb: fb.append("✅ 动作轨迹良好，继续保持！")
    return fb

def gen_thumbnail(vpath, opath):
    try:
        ffmpeg.input(str(vpath),ss=1).filter('scale',160,120)\
            .output(str(opath),vframes=1,format='image2',vcodec='mjpeg')\
            .overwrite_output().run(capture_stdout=True,capture_stderr=True)
        return True
    except: return False


# ================== 4. 核心接口 ==================
@app.post("/analyze-barbell")
async def analyze_barbell(video: UploadFile = File(...)):
    aid = str(uuid.uuid4())
    tdir = UPLOADS_DIR / aid; tdir.mkdir(exist_ok=True)
    
    try:
        vpath = tdir / video.filename
        with open(vpath,"wb") as f: shutil.copyfileobj(video.file, f)
        
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened(): raise HTTPException(400,"无法打开视频")
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print(f"📊 {fps:.0f}fps {fc}帧 {fw}x{fh}")
        
        trajectory, pose_data, fidx = [], [], 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            lms = None
            if MEDIAPIPE_AVAILABLE:
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mpi = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    ts = int(fidx * (1000/fps))
                    det = pose_landmarker.detect_for_video(mpi, ts)
                    lms = extract_landmarks(det)
                    
                    if lms:
                        metrics = compute_pose_metrics(lms, fw, fh)
                        if metrics:
                            metrics['frame'] = fidx
                            pose_data.append(metrics)
                except Exception as e:
                    if fidx % 100 == 0: print(f"⚠️ MP帧{fidx}: {e}")
            
            bp = detect_barbell_position(frame, lms)
            if bp: trajectory.append([bp[0], bp[1]])
            
            fidx += 1
            if fidx % 30 == 0: print(f"⏳ {fidx}/{fc}")
        
        cap.release()
        
        tp = tdir/f"{aid}_thumb.jpg"
        tu = f"/uploads/{aid}/{aid}_thumb.jpg" if gen_thumbnail(vpath,tp) and tp.exists() else None
        video_url = f"/uploads/{aid}/{video.filename}"
        
        if not is_valid_video(trajectory, pose_data):
            return {"analysis_id": aid, "thumbnailUrl": tu, 
                    "videoUrl": video_url,
                    "analysis_time": datetime.now().isoformat(), 
                    **UNKNOWN_RESPONSE}
        
        ex_type = RobustExerciseClassifier.classify(trajectory, pose_data, fh)
        
        if ex_type == "Unknown":
            return {"analysis_id": aid, "thumbnailUrl": tu, 
                    "videoUrl": video_url,
                    "analysis_time": datetime.now().isoformat(), 
                    **UNKNOWN_RESPONSE}
        
        ta = analyze_trajectory(trajectory)
        rpe = calc_rpe(ta, pose_data)
        ss = float(ta['stability'].replace('%',''))
        score = min(100, int(ss*0.7 + (10-rpe+5)*7))
        fb = gen_feedback(ta, pose_data, ex_type)
        
        print(f"\n✅ {ex_type} | 分:{score} | RPE:{rpe}")
        return {"success":True,"analysis_id":aid,"exercise_type":ex_type,"score":score,
                "stability":ta['stability'],"offset":ta['offset'],
                "avg_speed":ta['avg_speed'],"max_speed":ta['max_speed'],
                "sticking_point":ta['sticking_point'],"rpe":rpe,"feedback":fb,
                "trajectory":trajectory[:100],"thumbnailUrl":tu,
                "videoUrl":video_url,
                "analysis_time":datetime.now().isoformat()}
                
    except Exception as e:
        print(f"❌ {e}"); import traceback; traceback.print_exc()
        shutil.rmtree(tdir, ignore_errors=True)
        raise HTTPException(500, f"分析失败: {e}")

@app.get("/health")
async def health():
    return {"status":"OK","timestamp":datetime.now().isoformat(),
            "models":{"yolo":"Available" if YOLO_AVAILABLE else "N/A",
                      "mediapipe":"Available" if MEDIAPIPE_AVAILABLE else "N/A"}}

if __name__ == "__main__":
    print("\n"+"="*60)
    print("🏋️ WL Analysis AI Backend  (卧推误杀修复版)")
    print("="*60+"\n")
    uvicorn.run(app, host="0.0.0.0", port=8001)