# main.py
import warnings
warnings.filterwarnings("ignore", message=".*SymbolDatabase.GetPrototype.*")

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


from app.api.routes import router
from app.core.config import UPLOADS_DIR

import math, json
from fastapi.responses import JSONResponse as _JSONResponse

class SafeJSONResponse(_JSONResponse):
    def render(self, content) -> bytes:
        def _sanitize(obj):
            if isinstance(obj, float):
                if math.isnan(obj) or math.isinf(obj):
                    return None
                return obj
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_sanitize(v) for v in obj]
            return obj
        return json.dumps(_sanitize(content), ensure_ascii=False).encode("utf-8")


app = FastAPI(default_response_class=SafeJSONResponse,title="WL Analysis AI Backend V10")

# 1. CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. 静态文件挂载 (必须在 include_router 之前！)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# 3. ⭐️ 注册路由，确保没有 prefix（或者 prefix=""）
app.include_router(router)  # ← 确保这里没有 prefix="/api"

@app.get("/health")
async def health_check():
    return {"status": "OK", "version": "10.0.0 (Modular)", "models": {"mediapipe": True, "yolo": True}}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)