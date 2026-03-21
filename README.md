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
  "blur_faces": true,
  "blur_background_faces": true,
  "preserve_primary_subjects": true,
  "primary_subject_count": 1,
  "blur_sensitive_text": false,
  "detect_nudity": true,
  "censor_sensitive_audio": true,
  "keep_audio": true,
  "trusted_face_threshold": 0.82,
  "nudity_threshold": 0.55,
  "nudity_sample_stride": 5,
  "temporal_smoothing_alpha": 0.7,
  "temporal_blur_threshold": 0.5
}
```

Queue response includes:

- `job_id`
- `status` (queued)
- `status_url` (poll endpoint)

### `GET /jobs/<job_id>`
Poll this endpoint until status is `completed` or `failed`.

Completed response includes:

- `processed_video_url`
- `download_video_url`
- `report` with blur counts, muted intervals, and processing metadata

## Notes

- If FFmpeg is missing, output video is still produced, but audio may be dropped.
- Nudity model files are downloaded by NudeNet on first usage.
- Current face identity matching is lightweight (fast and practical), not face-recognition-grade biometrics.
