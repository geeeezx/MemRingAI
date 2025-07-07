# MemRingAI Transcription Service

A fast, async FastAPI service for audio transcription using OpenAI's Whisper API.

## Features

- 🚀 **Fast & Async**: Built with FastAPI for high-performance async processing
- 🎵 **Multiple Formats**: Supports MP3, MP4, MPEG, MPGA, M4A, WAV, WebM, OPUS
- 🌐 **URL Support**: Transcribe audio files from URLs
- 🤖 **Multiple ASR Providers**: Support for OpenAI Whisper and Volcengine (DouBao) ASR
- 🔄 **Provider Switching**: Seamlessly switch between different ASR providers
- 🎤 **Voice Activity Detection**: Pre-processing with Silero VAD for better transcription quality
- ⚡ **Audio Acceleration**: Optional audio speed-up for VAD segments to reduce processing time
- 🔄 **Format Conversion**: Automatic OPUS to WAV conversion for compatibility
- 📊 **Detailed Output**: Get timestamps, segments, and confidence scores
- 🔧 **Configurable**: Customizable model, language, and response formats
- 🛡️ **Secure**: File validation, size limits, and proper error handling
- 📚 **Documented**: Auto-generated API documentation with Swagger UI

## Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API key (for OpenAI Whisper)
- Volcengine (DouBao) credentials (for Volcengine ASR)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd MemRingAI
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```

3. **Set up environment variables**
   ```bash
   cp env.example .env
   # Edit .env and add your OpenAI API key
   ```

4. **Download VAD model (optional)**
   ```bash
   python download_vad_model.py
   ```

5. **Run the service**
   ```bash
   uv run python -m app.main
   ```

The service will be available at `http://localhost:8000`

## API Endpoints

### Health Check
```http
GET /api/v1/health
```

### Transcribe File
```http
POST /api/v1/transcribe
Content-Type: multipart/form-data

file: [audio file]
model: whisper-1 (optional)
language: en (optional)
prompt: "Optional context prompt" (optional)
response_format: verbose_json (optional)
temperature: 0.0 (optional)
provider: auto (optional) - ASR provider (openai, volcengine, auto)
enable_vad: true (optional) - Enable Voice Activity Detection
enable_acceleration: false (optional) - Enable audio acceleration for VAD segments
acceleration_factor: 1.5 (optional) - Audio acceleration factor (1.0-3.0)
min_segment_duration: 1.0 (optional) - Minimum segment duration for acceleration (seconds)
```

### Transcribe from URL
```http
POST /api/v1/transcribe/url
Content-Type: application/x-www-form-urlencoded

url: https://example.com/audio.mp3
model: whisper-1 (optional)
language: en (optional)
prompt: "Optional context prompt" (optional)
response_format: verbose_json (optional)
temperature: 0.0 (optional)
```

### Get Supported Formats
```http
GET /api/v1/supported-formats
```

### Get Available Providers
```http
GET /api/v1/providers
```

## Usage Examples

### Using curl

**Transcribe a local file:**
```bash
curl -X POST "http://localhost:8000/api/v1/transcribe" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.mp3" \
  -F "language=en" \
  -F "response_format=verbose_json" \
  -F "provider=openai"
```

**Transcribe from URL:**
```bash
curl -X POST "http://localhost:8000/api/v1/transcribe/url" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "url=https://example.com/audio.mp3&language=en"
```

### Using Python

```python
import requests

# Transcribe file
with open('audio.mp3', 'rb') as f:
    files = {'file': f}
    data = {
        'language': 'en',
        'response_format': 'verbose_json'
    }
    response = requests.post(
        'http://localhost:8000/api/v1/transcribe',
        files=files,
        data=data
    )
    result = response.json()
    print(result['text'])

# Transcribe from URL
data = {
    'url': 'https://example.com/audio.mp3',
    'language': 'en'
}
response = requests.post(
    'http://localhost:8000/api/v1/transcribe/url',
    data=data
)
result = response.json()
print(result['text'])
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | Required for OpenAI |
| `OPENAI_ORGANIZATION` | OpenAI organization ID | Optional |
| `VOLCENGINE_APP_ID` | Your Volcengine App ID | Required for Volcengine |
| `VOLCENGINE_ACCESS_TOKEN` | Your Volcengine Access Token | Required for Volcengine |
| `VOLCENGINE_RESOURCE_ID` | Volcengine Resource ID | Optional (default: volc.bigasr.auc_turbo) |
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8000` |
| `DEBUG` | Debug mode | `false` |
| `MAX_FILE_SIZE` | Max file size in bytes | `26214400` (25MB) |
| `ALLOWED_AUDIO_EXTENSIONS` | Allowed file extensions | `[".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm", ".opus"]` |
| `DEFAULT_WHISPER_MODEL` | Default Whisper model | `whisper-1` |
| `DEFAULT_LANGUAGE` | Default language | None |
| `DEFAULT_RESPONSE_FORMAT` | Default response format | `verbose_json` |
| `TEMP_DIR` | Temporary file directory | `./temp` |

### Voice Activity Detection (VAD)

The service includes Voice Activity Detection using the Silero VAD model to improve transcription quality:

- **Automatic Format Conversion**: OPUS files are automatically converted to WAV format
- **Speech Segmentation**: Detects speech segments and removes silence
- **Configurable Parameters**: Adjustable threshold, minimum speech duration, and silence duration
- **Optional Feature**: Can be enabled/disabled via the `enable_vad` parameter

**VAD Response Information:**
```json
{
  "vad_info": {
    "segment_count": 5,
    "speech_ratio": 0.75,
    "total_speech_duration": 45.2,
    "speech_segments": [
      {"start": 0.0, "end": 10.5, "confidence": 1.0, "accelerated": true},
      {"start": 15.2, "end": 25.8, "confidence": 1.0, "accelerated": false}
    ],
    "acceleration_info": {
      "applied": true,
      "original_duration": 60.0,
      "accelerated_duration": 40.0,
      "factor": 1.5
    }
  }
}
```

### Audio Acceleration

The service supports optional audio acceleration for VAD segments to reduce processing time:

- **Selective Acceleration**: Only segments longer than a minimum duration are accelerated
- **Configurable Speed**: Adjustable acceleration factor from 1.0x to 3.0x
- **Quality Preservation**: Short segments keep original speed to maintain quality
- **Time Savings**: Can reduce audio duration by 20-50% depending on settings

**Acceleration Parameters:**
- `enable_acceleration`: Enable/disable audio acceleration (default: false)
- `acceleration_factor`: Speed multiplier (1.0 = normal, 2.0 = 2x speed, default: 1.5)
- `min_segment_duration`: Minimum segment duration for acceleration in seconds (default: 1.0)

**Example Usage:**
```bash
curl -X POST "http://localhost:8000/api/v1/transcribe" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.mp3" \
  -F "enable_vad=true" \
  -F "enable_acceleration=true" \
  -F "acceleration_factor=1.8" \
  -F "min_segment_duration=1.5"
```

### Response Formats

- `verbose_json`: Detailed JSON with segments and timestamps
- `json`: Simple JSON with text only
- `text`: Plain text
- `srt`: SubRip subtitle format
- `vtt`: WebVTT subtitle format

## Development

### Running in Development Mode

```bash
uv run python -m app.main
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Test transcription with different models
python tests/test_simple.py tests/audio_test/ref_lbw.WAV gpt-4o-transcribe

# Test VAD functionality
python tests/test_vad.py tests/audio_test/ref_lbw.WAV true zh
```

### Code Formatting

```bash
uv run black .
uv run isort .
```

## API Documentation

Once the service is running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Project Structure

```
MemRingAI/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration settings
│   ├── models.py            # Pydantic models
│   ├── api/
│   │   ├── __init__.py
│   │   └── transcription.py # API endpoints
│   └── services/
│       ├── __init__.py
│       ├── openai_service.py # OpenAI API integration
│       └── file_service.py   # File handling
├── pyproject.toml           # Project configuration
├── env.example              # Environment variables example
└── README.md               # This file
```

## Error Handling

The service includes comprehensive error handling:

- **400 Bad Request**: Invalid file type, malformed request
- **413 Payload Too Large**: File exceeds size limit
- **500 Internal Server Error**: OpenAI API errors, file processing errors

## Security Considerations

- File type validation
- File size limits
- Temporary file cleanup
- CORS configuration (customize for production)
- Environment variable validation

## License

[Add your license here]
