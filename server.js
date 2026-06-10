const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs').promises;
const fsSync = require('fs');
const cors = require('cors');
const { v4: uuidv4 } = require('uuid');
const ffmpeg = require('fluent-ffmpeg');

ffmpeg.setFfmpegPath('D:\\data\\ffmpeg-8.1.1-full_build\\ffmpeg-8.1.1-full_build\\bin\\ffmpeg.exe');
ffmpeg.setFfprobePath('D:\\data\\ffmpeg-8.1.1-full_build\\ffmpeg-8.1.1-full_build\\bin\\ffprobe.exe');

// 初始化 Express 应用
const app = express();
const PORT = process.env.PORT || 8000;

// 确保目录存在
const UPLOADS_DIR = path.join(__dirname, 'uploads');
const CHUNKS_DIR = path.join(__dirname, 'chunks');

async function ensureDirectories() {
  try {
    await fs.mkdir(UPLOADS_DIR, { recursive: true });
    await fs.mkdir(CHUNKS_DIR, { recursive: true });
    console.log('Directories ready');
  } catch (error) {
    console.error('Directory error:', error);
  }
}

// 中间件 - 简化配置
app.use(cors());
app.use(express.json({ limit: '50mb' }));

// 存储上传会话信息
const uploadSessions = new Map();

app.get('/health', (req, res) => {
  res.json({ status: 'OK', message: 'Server is running' });
});

// === 分块上传路由 ===
app.post('/init-upload', async (req, res) => {
  console.log('Received init-upload request');
  try {
    const { fileName, fileSize, totalChunks } = req.body;
    
    if (!fileName || !fileSize || !totalChunks) {
      return res.status(400).json({ 
        error: 'Missing required parameters',
        success: false
      });
    }

    const sessionId = uuidv4();
    const uniqueFileName = `${sessionId}-${fileName}`;
    
    uploadSessions.set(sessionId, {
      fileName: uniqueFileName,
      fileSize: parseInt(fileSize),
      totalChunks: parseInt(totalChunks),
      uploadedChunks: new Set(),
      createdAt: Date.now()
    });

    console.log(`Upload session created: ${sessionId}`);
    res.json({ 
      success: true, 
      sessionId,
      fileName: uniqueFileName
    });
  } catch (error) {
    console.error('Init upload error:', error);
    res.status(500).json({ 
      error: 'Failed to initialize upload',
      success: false
    });
  }
});

app.post('/upload-chunk', async (req, res) => {
  console.log('Received upload-chunk request');
  try {
    const { sessionId, chunkIndex, chunkData } = req.body;
    
    if (!sessionId || chunkIndex === undefined || !chunkData) {
      return res.status(400).json({ 
        error: 'Missing required parameters',
        success: false
      });
    }

    const session = uploadSessions.get(sessionId);
    if (!session) {
      return res.status(404).json({ 
        error: 'Upload session not found',
        success: false
      });
    }

    const chunkIndexNum = parseInt(chunkIndex);
    const chunkPath = path.join(CHUNKS_DIR, `${session.fileName}.part${chunkIndexNum}`);
    await fs.writeFile(chunkPath, Buffer.from(chunkData, 'base64'));

    session.uploadedChunks.add(chunkIndexNum);
    console.log(`Chunk ${chunkIndexNum} uploaded for session ${sessionId}`);

    res.json({ 
      success: true, 
      message: `Chunk ${chunkIndexNum} uploaded successfully`
    });
  } catch (error) {
    console.error('Upload chunk error:', error);
    res.status(500).json({ 
      error: 'Failed to upload chunk',
      success: false
    });
  }
});

// === 修改 merge-and-analyze 路由 ===
app.post('/merge-and-analyze', async (req, res) => {
  console.log('Received merge-and-analyze request');
  try {
    const { sessionId } = req.body;
    
    if (!sessionId) {
      return res.status(400).json({ 
        error: 'Missing session ID',
        success: false
      });
    }

    const session = uploadSessions.get(sessionId);
    if (!session) {
      return res.status(404).json({ 
        error: 'Upload session not found',
        success: false
      });
    }

    if (session.uploadedChunks.size !== session.totalChunks) {
      return res.status(400).json({ 
        error: `Incomplete upload: ${session.uploadedChunks.size}/${session.totalChunks} chunks`,
        success: false
      });
    }

    // 合并文件
    const outputPath = path.join(UPLOADS_DIR, session.fileName);
    const writeStream = fsSync.createWriteStream(outputPath);
    
    for (let i = 0; i < session.totalChunks; i++) {
      const chunkPath = path.join(CHUNKS_DIR, `${session.fileName}.part${i}`);
      const chunkData = await fs.readFile(chunkPath);
      writeStream.write(chunkData);
      await fs.unlink(chunkPath);
    }
    
    writeStream.end();

    // 清理会话
    uploadSessions.delete(sessionId);

    // === 新增：生成缩略图 ===
    const thumbnailPath = path.join(UPLOADS_DIR, `${session.fileName}_thumb.jpg`);
    
    // 使用 ffmpeg 提取第一帧作为缩略图
    await new Promise((resolve, reject) => {
      ffmpeg(outputPath)
        .on('end', resolve)
        .on('error', reject)
        .screenshots({
          count: 1,
          folder: UPLOADS_DIR,
          filename: `${session.fileName}_thumb.jpg`,
          size: '120x80'
        });
    });

    console.log('Thumbnail generated:', thumbnailPath);

    // 模拟分析（实际项目中这里会调用真正的 AI 分析）
    const mockResults = {
      success: true,
      analysis_id: uuidv4(),
      exercise_type: ['Bench Press', 'Squat', 'Deadlift'][Math.floor(Math.random() * 3)],
      score: Math.floor(80 + Math.random() * 20),
      stability: `${(80 + Math.random() * 20).toFixed(1)}%`,
      offset: Math.random() > 0.5 
        ? `左偏 ${(1 + Math.random() * 5).toFixed(1)}cm` 
        : `右偏 ${(1 + Math.random() * 5).toFixed(1)}cm`,
      message: 'AI 分析完成！',
      // === 新增：返回缩略图路径 ===
      thumbnailUrl: `/uploads/${session.fileName}_thumb.jpg`
    };

    res.json(mockResults);

  } catch (error) {
    console.error('Merge error:', error);
    res.status(500).json({ 
      error: 'Failed to merge and analyze',
      success: false
    });
  }
});

// === 新增：提供静态文件服务 ===
app.use('/uploads', express.static(UPLOADS_DIR));

// 启动服务器
ensureDirectories().then(() => {
  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server running on http://0.0.0.0:${PORT}`);
    console.log('Available endpoints:');
    console.log(`GET  /health`);
    console.log(`POST /init-upload`);
    console.log(`POST /upload-chunk`);
    console.log(`POST /merge-and-analyze`);
  });
});