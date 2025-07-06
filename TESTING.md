# MemRingAI 转录API测试指南

## 测试脚本说明

我们提供了多种测试方式来验证转录API的功能：

### 1. 使用Python脚本测试

#### 完整测试脚本 (`test_api.py`)
```bash
# 安装requests包（如果还没有安装）
uv add requests

# 运行完整测试
python test_api.py
```

#### 简单测试脚本 (`test_simple.py`)
```bash
# 测试单个音频文件
python test_simple.py path/to/your/audio.mp3
```

### 2. 使用curl测试 (Windows)

#### 使用批处理文件
```bash
# 测试单个音频文件
test_curl.bat your_audio_file.mp3
```

#### 手动curl命令
```bash
# 健康检查
curl -X GET "http://localhost:8000/api/v1/health"

# 获取支持格式
curl -X GET "http://localhost:8000/api/v1/supported-formats"

# 转录音频文件
curl -X POST "http://localhost:8000/api/v1/transcribe" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_audio.mp3" \
  -F "language=zh" \
  -F "model=whisper-1" \
  -F "response_format=verbose_json"
```

### 3. 使用浏览器测试

访问 http://localhost:8000/docs 查看交互式API文档，可以直接在浏览器中测试API。

## 支持的音频格式

- MP3 (.mp3)
- MP4 (.mp4)
- MPEG (.mpeg)
- MPGA (.mpga)
- M4A (.m4a)
- WAV (.wav)
- WebM (.webm)

## 测试步骤

### 1. 确保服务正在运行
```bash
# 启动服务
uv run run.py
```

### 2. 准备测试音频文件
- 将音频文件放在项目目录中
- 确保文件格式受支持
- 文件大小不超过25MB

### 3. 运行测试
选择以下任一方式：

#### 方式A: Python脚本
```bash
python test_simple.py your_audio.mp3
```

#### 方式B: curl批处理
```bash
test_curl.bat your_audio.mp3
```

#### 方式C: 浏览器
1. 打开 http://localhost:8000/docs
2. 点击 `/api/v1/transcribe` 端点
3. 点击 "Try it out"
4. 上传音频文件
5. 点击 "Execute"

## 测试参数说明

### 必需参数
- `file`: 音频文件

### 可选参数
- `language`: 语言代码 (zh=中文, en=英文, ja=日文等)
- `model`: Whisper模型 (默认: whisper-1)
- `response_format`: 响应格式 (默认: verbose_json)
- `temperature`: 采样温度 (默认: 0.0)
- `prompt`: 可选提示词

## 响应格式

### verbose_json (默认)
```json
{
  "task": "transcribe",
  "language": "zh",
  "duration": 10.5,
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 2.5,
      "text": "转录的文本内容",
      "tokens": [1, 2, 3],
      "temperature": 0.0,
      "avg_logprob": -0.5,
      "compression_ratio": 1.0,
      "no_speech_prob": 0.1
    }
  ],
  "text": "完整的转录文本"
}
```

### 其他格式
- `json`: 仅包含文本
- `text`: 纯文本
- `srt`: 字幕格式
- `vtt`: WebVTT格式

## 故障排除

### 常见问题

1. **服务未启动**
   - 确保运行 `uv run run.py`
   - 检查端口8000是否被占用

2. **文件格式不支持**
   - 检查文件扩展名是否在支持列表中
   - 转换文件格式

3. **文件过大**
   - 默认限制25MB
   - 压缩音频文件或分割文件

4. **API密钥未配置**
   - 确保在 `.env` 文件中设置了 `OPENAI_API_KEY`
   - 检查API密钥是否有效

5. **网络连接问题**
   - 检查防火墙设置
   - 确保能访问OpenAI API

### 调试信息

查看服务日志获取详细错误信息：
```bash
# 启动服务时查看日志
uv run run.py
```

## 示例音频文件

如果您没有测试音频文件，可以：

1. 录制一段语音
2. 下载公开的音频文件
3. 使用在线文本转语音服务生成测试文件

## 性能测试

对于大文件或批量测试，建议：

1. 使用较小的音频文件进行初步测试
2. 逐步增加文件大小
3. 监控API响应时间
4. 注意OpenAI API的使用限制和费用 