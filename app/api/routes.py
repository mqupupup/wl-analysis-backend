# app/api/routes.py

from fastapi import APIRouter, HTTPException, Form
from pydantic import BaseModel, Field, ConfigDict
from typing import List
import traceback
from app.services.biomechanics_service import analyze as biomechanics_analyze

from app.services import upload_service
from app.services.v2_analysis_service import run_v2_analysis
router = APIRouter()


class InitUploadRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    file_name: str = Field(..., alias="fileName")
    file_size: int = Field(..., alias="fileSize")
    total_chunks: int = Field(..., alias="totalChunks")


class UploadChunkRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    session_id: str = Field(..., alias="sessionId")
    chunk_index: int = Field(..., alias="chunkIndex")
    chunk_data: str = Field(..., alias="chunkData")


@router.post("/init-upload")
async def init_upload(request: InitUploadRequest):
    """初始化分块上传会话"""
    try:
        print(f"[UPLOAD] Init: {request.file_name}, size: {request.file_size}, chunks: {request.total_chunks}")
        
        result = upload_service.init_upload_session(
            file_name=request.file_name,
            file_size=request.file_size,
            total_chunks=request.total_chunks
        )
        
        return {
            "success": True,
            "sessionId": result["sessionId"],
            "chunkSize": result.get("chunkSize", 5 * 1024 * 1024)
        }
    
    except Exception as e:
        print(f"[ERROR] Init upload failed: {str(e)}")
        traceback.print_exc()
        return {"success": False, "error": f"初始化上传失败: {str(e)}"}


@router.get("/get-uploaded-chunks/{session_id}")
async def get_uploaded_chunks(session_id: str):
    """查询已上传的分块列表（断点续传用）"""
    try:
        chunks = upload_service.get_uploaded_chunks(session_id)
        return {
            "success": True,
            "uploadedChunks": chunks
        }
    except Exception as e:
        print(f"[ERROR] Query chunks failed: {str(e)}")
        traceback.print_exc()
        return {
            "success": True,
            "uploadedChunks": []
        }


@router.post("/upload-chunk")
async def upload_chunk(request: UploadChunkRequest):
    """上传单个分块"""
    try:
        result = upload_service.save_chunk(
            session_id=request.session_id,
            chunk_index=request.chunk_index,
            chunk_data=request.chunk_data
        )
        return {"success": True, **result}
    
    except Exception as e:
        print(f"[ERROR] Upload chunk failed: {str(e)}")
        traceback.print_exc()
        raise HTTPException(500, f"上传分块失败: {str(e)}")


class MergeAnalyzeRequest(BaseModel):
    """与旧版保持一致，接收 JSON 格式的 sessionId"""
    model_config = ConfigDict(populate_by_name=True)
    session_id: str = Field(..., alias="sessionId")


@router.post("/merge-and-analyze")
async def merge_and_analyze(request: MergeAnalyzeRequest):
    """合并分块并触发分析"""
    try:
        session_id = request.session_id
        print(f"[MERGE] Starting merge: {session_id}")
        
        # 1. 合并视频分块
        video_path = upload_service.merge_chunks(session_id)
        
        # 2. 生成缩略图并记录相对URL路径
        thumb_filename = "thumbnail.jpg"
        upload_service.generate_thumbnail(video_path, thumb_filename)
        thumbnail_url = f"/uploads/{session_id}/{thumb_filename}"
        
        # 3. ✅ 核心修复：只传 video_path 一个参数，去掉多余的参数
        print(f"[ANALYSIS] Starting video analysis: {video_path}")
        # analysis_result = analysis_service.analyze(video_path)
        analysis_result = biomechanics_analyze(video_path)
        
        # 4. 返回结果，前端期望 exercise_type / thumbnailUrl 等在根级别
        video_url = f"/uploads/{session_id}/{video_path.name}"
        return {
            "success": True,
            "thumbnailUrl": thumbnail_url,
            "video_url": video_url,
            **analysis_result
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Merge and analyze failed: {str(e)}")
        traceback.print_exc()
        return {"success": False, "error": f"合并分析失败: {str(e)}"}


@router.get("/health")
async def health_check():
    return {"status": "ok", "message": "V10 Backend is running"}