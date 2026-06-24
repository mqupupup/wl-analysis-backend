import cv2
import numpy as np
import mediapipe as mp
from enum import IntEnum

# ================== 1. 关键点定义 (MediaPipe Pose 33点标准) ==================
class PoseLandmark(IntEnum):
    """MediaPipe Pose 33个关键点索引（与用户问题完全一致）"""
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32

# ================== 2. 核心工具函数 ==================
def calculate_angle(a, b, c):
    """计算三点构成的夹角（单位：度）"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180 else 360 - angle

def detect_barbell_trajectory(frame, prev_point=None):
    """
    通过颜色阈值检测杠铃轨迹（假设杠铃片为深色）
    实际应用中可替换为更鲁棒的跟踪算法（如KCF）
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 深色杠铃片的HSV范围（需根据实际视频调整）
    lower_dark = np.array([0, 0, 20])
    upper_dark = np.array([180, 255, 80])
    mask = cv2.inRange(hsv, lower_dark, upper_dark)
    
    # 形态学操作降噪
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 寻找最大轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 500:  # 最小面积过滤
            M = cv2.moments(largest)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy)
    return prev_point  # 无检测时沿用上一帧位置

# ================== 3. 动作分类器（终极修复版） ==================
def classify_exercise(trajectory, pose_data, frame_height):
    """
    根据轨迹和姿态数据分类三大动作（引入状态机和一票否决机制）
    """
    if not trajectory or len(trajectory) < 15:
        return "Unknown"
    
    # 1. 轨迹特征分析
    y_coords = [point[1] for point in trajectory]
    y_range = max(y_coords) - min(y_coords)
    x_coords = [point[0] for point in trajectory]
    x_std = np.std(x_coords)
    
    relative_y_range = (y_range / frame_height) * 100
    relative_x_std = (x_std / frame_height) * 100
    
    # 2. 姿态特征分析（关键！）
    has_pose_data = bool(pose_data)
    if not has_pose_data:
        return "Unknown"
    
    # 提取所有帧的肘角和髋角
    elbow_angles = [d['elbow_angle'] for d in pose_data]
    hip_angles = [d['hip_angle'] for d in pose_data]
    
    min_elbow_angle = np.min(elbow_angles)
    hip_range = max(hip_angles) - min(hip_angles)
    
    # ========== 核心分类逻辑 ==========
    # 一票否决：肘角 < 90° → 绝对是卧推（深蹲/硬拉手臂必须伸直）
    if min_elbow_angle < 90:
        return "Bench Press"
    
    # 深蹲判定：髋角大幅变化 + 手臂伸直 + 大垂直位移
    if hip_range > 30 and min_elbow_angle > 120 and relative_y_range > 25:
        return "Squat"
    
    # 硬拉判定：起始位置低 + 手臂极度伸直 + 水平稳定
    if min_elbow_angle > 140 and relative_y_range > 20 and relative_x_std < 8:
        start_y_ratio = (y_coords[0] / frame_height) * 100
        if start_y_ratio > 55:  # 硬拉起始点通常在画面下半部
            return "Deadlift"
    
    # 卧推次级判定：髋角稳定 + 中等垂直位移
    if hip_range < 20 and 15 < relative_y_range < 35:
        return "Bench Press"
    
    return "Unknown"  # 无法明确分类

# ================== 4. 主程序 ==================
def main(video_path=None):
    """主分析流程"""
    # 初始化MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,  # 平衡速度与精度
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # 视频输入
    cap = cv2.VideoCapture(video_path if video_path else 0)
    if not cap.isOpened():
        print("❌ 无法打开视频源")
        return
    
    # 数据缓存
    trajectory = []      # 杠铃轨迹 [(x,y), ...]
    pose_data = []       # 姿态特征 [{'elbow_angle': , 'hip_angle': }, ...]
    current_exercise = "Unknown"
    frame_count = 0
    
    print("✅ 系统就绪！正在分析动作...\n按 ESC 退出")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1
        frame_height = frame.shape[0]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # === 步骤1：检测杠铃轨迹 ===
        barbell_pos = detect_barbell_trajectory(frame)
        if barbell_pos:
            trajectory.append(barbell_pos)
            cv2.circle(frame, barbell_pos, 8, (0, 0, 255), -1)  # 红色轨迹点
        
        # === 步骤2：检测人体姿态 ===
        pose_results = pose.process(rgb_frame)
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            
            try:
                # 左侧关键点（避免镜像问题，统一用左侧）
                l_shoulder = [landmarks[PoseLandmark.LEFT_SHOULDER].x, landmarks[PoseLandmark.LEFT_SHOULDER].y]
                l_elbow = [landmarks[PoseLandmark.LEFT_ELBOW].x, landmarks[PoseLandmark.LEFT_ELBOW].y]
                l_wrist = [landmarks[PoseLandmark.LEFT_WRIST].x, landmarks[PoseLandmark.LEFT_WRIST].y]
                l_hip = [landmarks[PoseLandmark.LEFT_HIP].x, landmarks[PoseLandmark.LEFT_HIP].y]
                l_knee = [landmarks[PoseLandmark.LEFT_KNEE].x, landmarks[PoseLandmark.LEFT_KNEE].y]
                
                # 计算关键角度
                elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
                hip_angle = calculate_angle(l_shoulder, l_hip, l_knee)  # 肩-髋-膝
                
                # 有效数据过滤
                if 10 <= elbow_angle <= 175 and 30 <= hip_angle <= 175:
                    pose_data.append({
                        'frame': frame_count,
                        'elbow_angle': elbow_angle,
                        'hip_angle': hip_angle
                    })
                    
                    # 实时显示角度
                    cv2.putText(frame, f"Elbow: {elbow_angle:.0f}°", 
                               (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Hip: {hip_angle:.0f}°", 
                               (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            except Exception as e:
                print(f"⚠️ 关键点计算失败: {str(e)}")
        
        # === 步骤3：每30帧分类一次 ===
        if frame_count % 30 == 0 and len(trajectory) > 15:
            current_exercise = classify_exercise(
                trajectory[-30:],  # 用最近1秒数据
                pose_data[-30:],
                frame_height
            )
        
        # === 步骤4：显示结果 ===
        cv2.putText(frame, f"Exercise: {current_exercise}", 
                   (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        
        # 绘制轨迹
        for i in range(1, len(trajectory)):
            if trajectory[i-1] and trajectory[i]:
                cv2.line(frame, trajectory[i-1], trajectory[i], (0, 255, 255), 2)
        
        # 显示视频
        cv2.imshow('Strength Analyzer', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC退出
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ 分析完成！")

if __name__ == "__main__":
    # 用法：python strength_analyzer.py [视频路径]
    import sys
    video_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(video_path)