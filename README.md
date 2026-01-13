# Context-Aware Video Censor Lab

A full-stack app where users upload a video and apply configurable privacy policies:

- Identity-aware face blurring (keep streamer/associates visible)
- Background face blurring
- Sensitive text-like region blurring
- Nudity detection and blur (NudeNet)
- Audio muting during sensitive events

## Novel Processing Approach

This project uses a hybrid strategy called **Context Graph Censor v1**:

1. Face detections are linked into lightweight tracks over time.
2. Tracks are scored by persistence, center-priority, and scene presence.
3. Trusted identities are preserved using reference-face similarity.
4. Face decisions are temporally smoothed to reduce blur flicker frame-to-frame.
5. Sensitive modalities (visual + audio) are censored selectively, not globally.

## Project Structure

- `backend/app.py`: Flask API endpoints and upload orchestration
- `backend/video_processing.py`: core censoring pipeline
- `backend/settings.py`: central configuration and default options
- `backend/options.py`: option normalization/validation
- `backend/media_utils.py`: reusable media/file helper utilities
- `backend/job_manager.py`: async queue/job state management
- `frontend/home.jsx`: main React UI
- `frontend/src/index.css`: visual design and responsive styles

## Engineering Principles Applied

- **Single Responsibility (SOLID)**: each backend module has one clear concern (config, validation, media utilities, job orchestration, API wiring, processing engine).
- **Open/Closed (SOLID)**: new processing options can be introduced in `settings.py` + `options.py` with minimal route-layer changes.
- **KISS**: the backend API remains thin and explicit; complex concerns are delegated to dedicated modules.
- **YAGNI**: avoided adding unnecessary infrastructure (database, distributed queues) while keeping interfaces ready for future extension.
- **Readability/Modularity**: functions are small, named by intent, and grouped by domain responsibility.

## Prerequisites

1. Python 3.10+
2. Node.js 18+
3. FFmpeg installed and available on PATH (recommended for audio preserve/censor)

## Backend Setup

```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Backend runs on: `http://localhost:5000`

## Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on: `http://localhost:3000`

## One-Click Scripts

From repository root:

- `start_backend.bat`
- `start_frontend.bat`

## API

### `POST /process-video`
Multipart form:

- `video`: uploaded video file
- `options`: JSON string of options
- `trusted_faces`: optional multiple image files (face references)

This endpoint is asynchronous and returns `202 Accepted` with a job id.

Sample options payload:

```json
{
  "message": "Video uploaded and processed successfully",
  "original_filename": "example.mp4",
  "temp_filename": "uuid.mp4",
  "temp_path": "/path/to/temp/file",
  "file_size": "10.5 MB",
  "upload_time": "2026-01-09T...",
  "video_info": {
    "width": 1920,
    "height": 1080,
    "fps": 30,
    "duration": 120.5,
    "frame_count": 3615
  },
  "processed_video_url": "/download-processed/processed_uuid.mp4",
  "preview_data": {
    "processing_applied": "Frame numbering and 10-second preview",
    "original_duration": 120.5,
    "preview_duration": 10
  }
}
```

### GET /download-processed/<filename>
Download a processed video file.

### GET /stream-processed/<filename>
Stream a processed video file for preview.

### GET /list-temp-files
List all temporary files with metadata.

### POST /cleanup
Manually trigger cleanup of expired files.

### GET /health
Health check endpoint.

## AI/ML Components

### Video Processor (`video_processor.py`)
- **YOLO Integration**: Object detection using YOLOv8 models
- **OCR Capabilities**: Text extraction using EasyOCR
- **GPU Acceleration**: Automatic GPU detection with CPU fallback
- **Frame Analysis**: Advanced frame-by-frame processing
- **Privacy Features**: Face blurring and text redaction capabilities

### LLM Manager (`llm_manager.py`)
- Intelligent content analysis and processing
- Natural language processing capabilities
- Extensible architecture for various LLM integrations

### Audio Processor (`audio_processor.py`)
- Audio extraction and analysis
- Sound processing capabilities
- Integrated with video processing pipeline

## Configuration

### Backend Configuration
- **MAX_FILE_SIZE**: 100MB (configurable in app.py)
- **ALLOWED_EXTENSIONS**: mp4, avi, mov, wmv, flv, webm, mkv
- **TEMP_FILE_EXPIRY_HOURS**: 24 hours (files auto-deleted)
- **Upload Directory**: `temp_storage/`
- **Processed Directory**: `processed_videos/`
- **GPU Support**: Automatic GPU detection for YOLO and OCR
- **AI Models**: YOLOv8n for object detection, EasyOCR for text recognition

### Frontend Configuration
- **Build System**: Vite for fast development and building
- **React Version**: React 19 with modern features
- **BACKEND_URL**: `http://localhost:5000` (configurable in home.jsx)
- **Development Port**: 3000 (configurable in vite.config.js)
- **Supported File Types**: Video files only
- **Max Upload Size**: 100MB

## Dependencies

### Backend Dependencies
```
flask                 # Web framework
flask-cors           # Cross-origin resource sharing
werkzeug             # WSGI utilities
opencv-python        # Computer vision library
pillow               # Image processing
ffmpeg-python        # Video processing
ultralytics          # YOLO models
easyocr              # Optical character recognition
```

### Frontend Dependencies
```
react ^19.0.0        # React framework
react-dom ^19.0.0    # React DOM rendering
axios ^1.6.0         # HTTP client
vite ^6.0.0          # Build tool and dev server
@vitejs/plugin-react # React plugin for Vite
```

## File Structure

```
SDP/
├── README.md                  # Project documentation
├── start_backend.bat         # Quick start script for backend
├── start_frontend.bat        # Quick start script for frontend
├── backend/
│   ├── app.py                # Main Flask application
│   ├── main.py               # Entry point for integrated system
│   ├── llm_manager.py        # LLM integration and management
│   ├── audio_processor.py    # Audio processing capabilities
│   ├── video_processor.py    # AI-powered video processing (YOLO + OCR)
│   ├── tests.py              # Unit tests
│   ├── requirements.txt      # Python dependencies
│   ├── temp_storage/         # Uploaded videos (auto-created)
│   └── processed_videos/     # Processed videos (auto-created)
├── frontend/
│   ├── index.html           # HTML entry point
│   ├── package.json         # Node.js dependencies
│   ├── vite.config.js       # Vite configuration
│   ├── home.jsx             # Main React component (legacy location)
│   ├── README.md            # Frontend-specific documentation
│   └── src/
│       ├── index.jsx        # React application entry point
│       └── index.css        # Stylesheet
```

## Video Processing Capabilities

The system now includes advanced AI-powered video processing features:

### Current Processing Pipeline
1. **Video Upload & Validation**: Secure file upload with type and size validation
2. **Metadata Extraction**: Video resolution, fps, duration, and frame count analysis
3. **AI-Powered Analysis**: 
   - Object detection using YOLO models
   - Text recognition using OCR
   - Frame-by-frame intelligent processing
4. **Privacy Protection**:
   - Automatic face detection and blurring
   - Text redaction capabilities
5. **Preview Generation**: Creates processed video previews with frame numbering
6. **Temporary Storage**: Automatic cleanup after processing

### Customization Options
- **GPU Acceleration**: Automatically detects and uses GPU when available
- **Model Selection**: Configurable YOLO models (currently using YOLOv8n)
- **Processing Parameters**: Customizable through the processor classes
- **Output Formats**: Flexible output configuration for different use cases

**To customize processing:**
- Modify the `VIDEO` class in `video_processor.py` for advanced video analysis
- Update the `LLM` class in `llm_manager.py` for intelligent content processing
- Extend the `AUDIO` class in `audio_processor.py` for audio analysis features

## Testing

Run backend tests:
```bash
cd backend
python tests.py
```

## System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **Node.js**: 16.x or higher  
- **RAM**: 4GB (8GB recommended for GPU processing)
- **Storage**: 2GB free space for dependencies

### Recommended for AI Features
- **GPU**: CUDA-compatible GPU for accelerated processing
- **RAM**: 8GB+ for large video files
- **CPU**: Multi-core processor for parallel processing

### Optional Dependencies
- **CUDA**: For GPU acceleration (NVIDIA GPUs)
- **FFmpeg**: System-wide installation for advanced video processing

## Security Considerations

- File type validation
- File size limits
- Secure filename generation
- Automatic cleanup of temporary files
- CORS enabled for development (configure for production)

## Production Deployment

For production deployment:

1. Set `app.run(debug=False)` in app.py
2. Configure proper CORS origins
3. Use a production WSGI server (e.g., Gunicorn)
4. Set up proper file storage (cloud storage recommended)
5. Implement authentication if required
6. Configure HTTPS

## Troubleshooting

### Common Issues

1. **AI Model Installation Issues**:
   ```bash
   # For YOLO models
   pip install ultralytics --upgrade
   
   # For OCR
   pip install easyocr --upgrade
   ```

2. **GPU Detection Problems**:
   - Ensure CUDA is installed for NVIDIA GPUs
   - Check GPU availability with: `python -c "import torch; print(torch.cuda.is_available())"`
   - System will automatically fallback to CPU if GPU unavailable

3. **OpenCV Installation Issues**:
   ```bash
   pip install opencv-python-headless
   ```

4. **FFmpeg Not Found**:
   - Install FFmpeg system-wide
   - Or use: `pip install imageio-ffmpeg`

5. **CORS Issues**:
   - Ensure backend is running on correct port (5000)
   - Check BACKEND_URL in frontend configuration
   - Verify frontend is running on port 3000

6. **File Upload Issues**:
   - Check file size (max 100MB)
   - Verify file format is supported
   - Ensure backend server is running

7. **Vite Development Server Issues**:
   ```bash
   # Clear cache and reinstall
   cd frontend
   rm -rf node_modules package-lock.json
   npm install
   ```

8. **Memory Issues with Large Videos**:
   - Reduce video file size or resolution
   - Increase system RAM
   - Enable GPU processing to reduce CPU memory usage

### Performance Optimization

- **Enable GPU**: Ensure CUDA is installed for faster processing
- **Batch Processing**: Process multiple videos in sequence rather than parallel
- **Memory Management**: Close browser tabs and other applications when processing large files
- **Storage**: Use SSD storage for faster I/O operations