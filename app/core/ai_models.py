import warnings
warnings.filterwarnings("ignore", message=".*SymbolDatabase.GetPrototype.*")

from pathlib import Path
import ssl
import urllib.request

MEDIAPIPE_AVAILABLE = False
YOLO_AVAILABLE = False
pose_landmarker = None
barbell_model = None

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    model_path = Path(__file__).parent.parent.parent / "models" / "pose_landmarker_heavy.task"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print("💡 正在下载 MediaPipe 模型...")
        ssl._create_default_https_context = ssl._create_unverified_context
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
            str(model_path)
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

try:
    from ultralytics import YOLO
    yolo_path = Path(__file__).parent.parent.parent / "models" / "yolov8n.pt"
    if yolo_path.exists():
        barbell_model = YOLO(str(yolo_path))
        YOLO_AVAILABLE = True
        print("✅ YOLOv8 加载成功")
    else:
        print("⚠️ YOLOv8 模型不存在 (不影响核心分析)")
except Exception as e:
    print(f"⚠️ YOLOv8 加载失败: {e}")