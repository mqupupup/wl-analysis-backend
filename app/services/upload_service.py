# app/services/upload_service.py

import base64
import shutil
import cv2
import uuid
from pathlib import Path
from typing import List, Dict, Any
from fastapi import HTTPException

# 确保 UPLOADS_DIR 是绝对路径
try:
    from app.core.config import UPLOADS_DIR
except ImportError:
    # 如果 config 里没有定义，使用默认值
    UPLOADS_DIR = Path(__file__).parent.parent.parent / "uploads"

# 确保目录存在
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
print(f"[UPLOAD] Upload directory: {UPLOADS_DIR.absolute()}")

# 内存存储上传会话（生产环境建议用 Redis）
upload_sessions: Dict[str, Dict[str, Any]] = {}


def init_upload_session(file_name: str, file_size: int, total_chunks: int) -> Dict[str, Any]:
    """初始化上传会话"""
    session_id = str(uuid.uuid4())
    
    # 创建会话目录
    session_dir = UPLOADS_DIR / session_id
    chunks_dir = session_dir / "chunks"
    
    try:
        chunks_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"[ERROR] Failed to create directory: {e}")
        raise HTTPException(500, f"无法创建上传目录: {str(e)}")
    
    # 记录会话信息
    upload_sessions[session_id] = {
        "file_name": file_name,
        "file_size": file_size,
        "total_chunks": total_chunks,
        "uploaded_chunks": set()
    }
    
    print(f"[UPLOAD] Init session: {session_id}, file: {file_name}")
    
    return {
        "sessionId": session_id,
        "chunkSize": 2 * 1024 * 1024  # 建议前端每块 2MB
    }


def save_chunk(session_id: str, chunk_index: int, chunk_data: str) -> Dict[str, Any]:
    """保存单个分块（Base64 编码的数据）"""
    if session_id not in upload_sessions:
        raise HTTPException(404, "上传会话不存在或已过期")
    
    session = upload_sessions[session_id]
    chunks_dir = UPLOADS_DIR / session_id / "chunks"
    
    # 解码 Base64 数据
    try:
        # 兼容前端可能带 data:application/octet-stream;base64, 前缀
        if ',' in chunk_data:
            chunk_data = chunk_data.split(',', 1)[1]
        
        binary_data = base64.b64decode(chunk_data)
    except Exception as e:
        raise HTTPException(400, f"分块数据解码失败: {str(e)}")
    
    # 写入文件
    chunk_path = chunks_dir / f"chunk_{chunk_index:04d}.bin"
    with open(chunk_path, "wb") as f:
        f.write(binary_data)
    
    # 记录已上传的分块
    session["uploaded_chunks"].add(chunk_index)
    
    return {
        "success": True,
        "chunkIndex": chunk_index,
        "uploadedChunks": len(session["uploaded_chunks"])
    }


def get_uploaded_chunks(session_id: str) -> List[int]:
    """查询某个会话下已上传的分块索引列表（断点续传用）"""
    if session_id not in upload_sessions:
        # 即使内存里没有记录，也尝试从磁盘恢复
        chunks_dir = UPLOADS_DIR / session_id / "chunks"
        if not chunks_dir.exists():
            return []
        
        chunks = []
        for f in chunks_dir.iterdir():
            if f.name.startswith("chunk_") and f.name.endswith(".bin"):
                try:
                    idx = int(f.name.replace("chunk_", "").replace(".bin", ""))
                    chunks.append(idx)
                except ValueError:
                    pass
        return sorted(chunks)
    
    return sorted(list(upload_sessions[session_id]["uploaded_chunks"]))


def merge_chunks(session_id: str) -> Path:
    """合并所有分块为最终视频文件"""
    if session_id not in upload_sessions:
        raise HTTPException(404, "上传会话不存在")
    
    session = upload_sessions[session_id]
    chunks_dir = UPLOADS_DIR / session_id / "chunks"
    output_dir = UPLOADS_DIR / session_id
    
    # 检查分块完整性
    if len(session["uploaded_chunks"]) != session["total_chunks"]:
        raise HTTPException(
            400, 
            f"分块不完整: 已上传 {len(session['uploaded_chunks'])}/{session['total_chunks']}"
        )
    
    # 合并文件
    output_path = output_dir / session["file_name"]
    with open(output_path, "wb") as outfile:
        for i in range(session["total_chunks"]):
            chunk_path = chunks_dir / f"chunk_{i:04d}.bin"
            if not chunk_path.exists():
                raise HTTPException(500, f"分块文件丢失: chunk_{i:04d}.bin")
            
            with open(chunk_path, "rb") as infile:
                shutil.copyfileobj(infile, outfile)
    
    # 清理分块目录
    shutil.rmtree(chunks_dir, ignore_errors=True)
    
    print(f"[OK] Chunks merged: {output_path}")
    
    # 清理内存会话
    del upload_sessions[session_id]
    
    return output_path


def generate_thumbnail(video_path: Path, thumb_name: str) -> Path:
    """从视频生成缩略图"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARNING] Cannot open video for thumbnail: {video_path}")
        return None
    
    # 读取第 30 帧（或第一帧）作为缩略图
    cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("[WARNING] Failed to read video frame")
        return None
    
    # 保存到会话目录
    thumb_path = video_path.parent / thumb_name
    cv2.imwrite(str(thumb_path), frame)
    
    print(f"[OK] Thumbnail generated: {thumb_path}")
    return thumb_path