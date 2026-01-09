# Video Upload & Preview System

A full-stack application for uploading videos, generating previews, and storing them temporarily.

## Features

- **Frontend (React)**:
  - Drag and drop video upload
  - Video preview before processing
  - Upload progress tracking
  - File validation and error handling
  - Responsive design
  
- **Backend (Flask)**:
  - Video file upload with validation
  - Temporary storage with automatic cleanup
  - Video processing (preview generation)
  - RESTful API endpoints
  - File streaming and download

## Setup Instructions

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   ```bash
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Start the backend server:
   ```bash
   python app.py
   ```

The backend server will start on `http://localhost:5000`

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

The frontend will be available at `http://localhost:3000`

## API Endpoints

### POST /upload-video
Upload and process a video file.

**Request:**
- Content-Type: multipart/form-data
- Body: video file in 'video' field

**Response:**
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

## Configuration

### Backend Configuration
- **MAX_FILE_SIZE**: 100MB (configurable in app.py)
- **ALLOWED_EXTENSIONS**: mp4, avi, mov, wmv, flv, webm, mkv
- **TEMP_FILE_EXPIRY_HOURS**: 24 hours (files auto-deleted)
- **Upload Directory**: `temp_storage/`
- **Processed Directory**: `processed_videos/`

### Frontend Configuration
- **BACKEND_URL**: `http://localhost:5000` (configurable in home.jsx)
- **Supported File Types**: Video files only
- **Max Upload Size**: 100MB

## File Structure

```
SDP/
├── backend/
│   ├── app.py                 # Main Flask application
│   ├── tests.py              # Unit tests
│   ├── requirements.txt      # Python dependencies
│   ├── temp_storage/         # Uploaded videos (auto-created)
│   └── processed_videos/     # Processed videos (auto-created)
├── frontend/
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── index.js          # React entry point
│   │   ├── index.css         # Styles
│   │   └── home.jsx          # Main component
│   └── package.json          # Node dependencies
└── README.md
```

## Video Processing

The current implementation includes a sample video processing pipeline that:

1. Extracts video metadata (resolution, fps, duration)
2. Creates a 10-second preview with frame numbering
3. Stores both original and processed videos temporarily

**To customize processing:**
Modify the `process_video_preview()` function in `app.py` to implement your specific video processing requirements.

## Testing

Run backend tests:
```bash
cd backend
python tests.py
```

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

1. **OpenCV Installation Issues**:
   ```bash
   pip install opencv-python-headless
   ```

2. **FFmpeg Not Found**:
   - Install FFmpeg system-wide
   - Or use: `pip install imageio-ffmpeg`

3. **CORS Issues**:
   - Ensure backend is running on correct port
   - Check BACKEND_URL in frontend configuration

4. **File Upload Issues**:
   - Check file size (max 100MB)
   - Verify file format is supported
   - Ensure backend server is running