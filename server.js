const express = require('express');
const path = require('path');
const fs = require('fs').promises;
const fsSync = require('fs');
const cors = require('cors');
const { v4: uuidv4 } = require('uuid');
const ffmpeg = require('fluent-ffmpeg');
const FormData = require('form-data');
const axios = require('axios'); // 新增

// 配置 ffmpeg 路径
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
    console.log('✅ Directories ready');
  } catch (error) {
    console.error('❌ Directory error:', error);
  }
}

// 中间件配置
app.use(cors({
  origin: '*',
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization']
}));

app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));

// 存储上传会话信息
const uploadSessions = new Map();

// 健康检查
app.get('/health', (req, res) => {
  res.json({ 
    status: 'OK', 
    message: 'Server is running',
    timestamp: new Date().toISOString(),
    uploadSessions: uploadSessions.size,
    nodeVersion: process.version
  });
});

// 初始化上传会话
app.post('/init-upload', async (req, res) => {
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

    console.log(`📁 Upload session created: ${sessionId}`);
    res.json({ 
      success: true, 
      sessionId,
      fileName: uniqueFileName
    });
  } catch (error) {
    console.error('❌ Init upload error:', error);
    res.status(500).json({ 
      error: 'Failed to initialize upload',
      success: false,
      details: error.message
    });
  }
});

// 上传分块
app.post('/upload-chunk', async (req, res) => {
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
    console.log(`📤 Chunk ${chunkIndexNum} uploaded for session ${sessionId}`);

    res.json({ 
      success: true, 
      message: `Chunk ${chunkIndexNum} uploaded successfully`,
      progress: `${session.uploadedChunks.size}/${session.totalChunks}`
    });
  } catch (error) {
    console.error('❌ Upload chunk error:', error);
    res.status(500).json({ 
      error: 'Failed to upload chunk',
      success: false,
      details: error.message
    });
  }
});

// 获取已上传分块
app.get('/get-uploaded-chunks/:sessionId', (req, res) => {
  const { sessionId } = req.params;
  
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

  res.json({
    success: true,
    uploadedChunks: Array.from(session.uploadedChunks),
    totalChunks: session.totalChunks
  });
});

// 合并分块并分析 - 修复版本（使用 axios）
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

    await new Promise((resolve) => {
      writeStream.on('finish', resolve);
      writeStream.on('error', resolve);
    });

    uploadSessions.delete(sessionId);

    console.log('✅ File merged successfully:', outputPath);

    // === 调用 FastAPI 后端进行真实分析 ===
    console.log('🚀 Calling FastAPI backend (http://localhost:8001/analyze-barbell)...');
    
    const formData = new FormData();
    const videoBuffer = await fs.readFile(outputPath);
    
    formData.append('video', videoBuffer, {
      filename: session.fileName,
      contentType: 'video/mp4'
    });

    const response = await axios.post(
      'http://localhost:8001/analyze-barbell',
      formData,
      {
        headers: {
          ...formData.getHeaders(),
          'Content-Type': `multipart/form-data; boundary=${formData.getBoundary()}`
        },
        maxContentLength: Infinity,
        maxBodyLength: Infinity,
        timeout: 300000
      }
    );

    console.log('✅ Analysis completed!');
    console.log('   Exercise Type:', response.data.exercise_type);
    console.log('   Score:', response.data.score);
    console.log('   Stability:', response.data.stability);

    res.json({
      success: true,
      analysis_id: response.data.analysis_id,
      exercise_type: response.data.exercise_type,
      score: response.data.score,
      stability: response.data.stability,
      offset: response.data.offset,
      avg_speed: response.data.avg_speed,
      max_speed: response.data.max_speed,
      sticking_point: response.data.sticking_point,
      rpe: response.data.rpe,
      feedback: response.data.feedback,
      thumbnailUrl: response.data.thumbnailUrl,
      videoUrl: response.data.videoUrl,
      trajectory: response.data.trajectory
    });

  } catch (error) {
    console.error('❌ Merge and analysis error:', error);
    
    let errorMessage = 'Failed to merge and analyze';
    if (error.response) {
      console.error('FastAPI error response:', error.response.data);
      errorMessage = `FastAPI error: ${JSON.stringify(error.response.data)}`;
    } else if (error.code === 'ECONNREFUSED') {
      errorMessage = '无法连接到 FastAPI 后端';
    } else {
      errorMessage = error.message;
    }
    
    res.status(500).json({ 
      error: errorMessage,
      success: false,
      details: error.message
    });
  }
});

// 静态文件服务
app.use('/uploads', express.static(UPLOADS_DIR, {
  maxAge: '1d',
  setHeaders: (res) => {
    res.setHeader('Cache-Control', 'public, max-age=86400');
  }
}));

// 启动服务器
ensureDirectories().then(() => {
  app.listen(PORT, '0.0.0.0', () => {
    console.log(`\n` + '='.repeat(60));
    console.log('🚀 Server running on http://0.0.0.0:' + PORT);
    console.log('='.repeat(60));
    console.log('\n📋 Available endpoints:');
    console.log('   GET  /health');
    console.log('   POST /init-upload');
    console.log('   POST /upload-chunk');
    console.log('   GET  /get-uploaded-chunks/:sessionId');
    console.log('   POST /merge-and-analyze');
    console.log('   GET  /uploads/* (static files)');
    console.log('\n🔧 Configuration:');
    console.log('   Node.js Version:', process.version);
    console.log('   FastAPI Backend: http://localhost:8001');
    console.log('   Uploads Directory:', UPLOADS_DIR);
    console.log('='.repeat(60) + '\n');
  });
});