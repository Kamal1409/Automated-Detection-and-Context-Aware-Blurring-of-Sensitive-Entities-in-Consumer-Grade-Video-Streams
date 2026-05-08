import json
import os
import threading
import time
import uuid
from datetime import datetime

from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from job_manager import JobManager
from face_candidates import extract_face_candidates
from media_utils import allowed_file, cleanup_expired_files, get_video_info
from options import normalize_options
from settings import (
    MAX_FILE_SIZE,
    PROCESSED_FOLDER,
    TEMP_FILE_EXPIRY_HOURS,
    TRUSTED_FOLDER,
    UPLOAD_FOLDER,
    ensure_directories,
)
from video_processing import process_video


app = Flask(__name__)
CORS(app)
ensure_directories()

app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

job_manager = JobManager(max_workers=2)


def _parse_options_payload():
    options_json = request.form.get("options")
    if not options_json:
        return normalize_options({})

    try:
        raw_options = json.loads(options_json)
    except json.JSONDecodeError as ex:
        raise ValueError(f"Invalid options payload: {ex}") from ex
    return normalize_options(raw_options)


def _save_trusted_faces(job_id):
    trusted_paths = []
    trusted_files = request.files.getlist("trusted_faces")
    if not trusted_files:
        return trusted_paths

    trusted_job_dir = os.path.join(TRUSTED_FOLDER, job_id)
    os.makedirs(trusted_job_dir, exist_ok=True)

    for trusted_file in trusted_files:
        if trusted_file.filename == "":
            continue
        safe_name = secure_filename(trusted_file.filename)
        file_path = os.path.join(trusted_job_dir, safe_name)
        trusted_file.save(file_path)
        trusted_paths.append(file_path)

    return trusted_paths


def _run_processing_job(
    job_id, input_path, output_path, original_name, output_name, options, trusted_paths
):
    job_manager.update(
        job_id,
        status="running",
        phase="initializing",
        progress=1.0,
        message="Preparing processing pipeline",
        started_at=datetime.now().isoformat(),
    )

    def on_progress(progress, message, phase):
        job_manager.update(
            job_id,
            status="running" if progress < 100 else "completed",
            phase=phase,
            progress=round(float(progress), 1),
            message=message,
        )

    try:
        report = process_video(
            input_path=input_path,
            output_path=output_path,
            options=options,
            trusted_face_paths=trusted_paths,
            progress_callback=on_progress,
        )

        result = {
            "message": "Video processed successfully",
            "job_id": job_id,
            "original_filename": original_name,
            "output_filename": output_name,
            "upload_time": datetime.now().isoformat(),
            "video_info": get_video_info(input_path),
            "effective_options": options,
            "report": report,
            "processed_video_url": f"/stream-processed/{output_name}",
            "download_video_url": f"/download-processed/{output_name}",
        }
        job_manager.complete(job_id, result)
    except Exception as ex:
        job_manager.fail(job_id, f"Processing failed: {ex}")


def _start_cleanup_scheduler():
    def cleanup_loop():
        while True:
            time.sleep(3600)
            try:
                cleanup_expired_files(
                    [UPLOAD_FOLDER, PROCESSED_FOLDER], TEMP_FILE_EXPIRY_HOURS
                )
            except Exception as ex:
                print(f"Cleanup error: {ex}")

    thread = threading.Thread(target=cleanup_loop, daemon=True)
    thread.start()


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})


@app.route("/process-video", methods=["POST"])
@app.route("/upload-video", methods=["POST"])
def process_video_endpoint():
    try:
        if "video" not in request.files:
            return jsonify({"error": "No video file provided"}), 400

        video_file = request.files["video"]
        if video_file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not allowed_file(video_file.filename):
            return (
                jsonify({"error": "Unsupported format. Use a standard video file."}),
                400,
            )

        options = _parse_options_payload()

        original_name = secure_filename(video_file.filename)
        extension = original_name.rsplit(".", 1)[1].lower()
        job_id = uuid.uuid4().hex

        input_name = f"input_{job_id}.{extension}"
        output_name = f"censored_{job_id}.mp4"
        input_path = os.path.join(UPLOAD_FOLDER, input_name)
        output_path = os.path.join(PROCESSED_FOLDER, output_name)

        video_file.save(input_path)
        trusted_paths = _save_trusted_faces(job_id)

        job_manager.create(
            job_id, message="Upload complete, waiting for available worker"
        )
        job_manager.submit(
            _run_processing_job,
            job_id,
            input_path,
            output_path,
            original_name,
            output_name,
            options,
            trusted_paths,
        )

        return (
            jsonify(
                {
                    "job_id": job_id,
                    "status": "queued",
                    "message": "Video accepted for async processing",
                    "status_url": f"/jobs/{job_id}",
                }
            ),
            202,
        )
    except ValueError as ex:
        return jsonify({"error": str(ex)}), 400
    except Exception as ex:
        return jsonify({"error": f"Processing failed: {ex}"}), 500


@app.route("/face-candidates", methods=["POST"])
def face_candidates_endpoint():
    try:
        if "video" not in request.files:
            return jsonify({"error": "No video file provided"}), 400

        video_file = request.files["video"]
        if video_file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not allowed_file(video_file.filename):
            return (
                jsonify({"error": "Unsupported format. Use a standard video file."}),
                400,
            )

        temp_name = f"facescan_{uuid.uuid4().hex}.{secure_filename(video_file.filename).rsplit('.', 1)[1].lower()}"
        temp_path = os.path.join(UPLOAD_FOLDER, temp_name)
        video_file.save(temp_path)

        try:
            max_candidates = int(request.form.get("max_candidates", 18))
            sample_stride = int(request.form.get("sample_stride", 2))
            max_seconds = int(request.form.get("max_seconds", 180))
            prefer_gpu = _parse_bool(request.form.get("prefer_gpu"), default=True)
            candidates = extract_face_candidates(
                temp_path,
                max_candidates=max(3, min(24, max_candidates)),
                sample_stride=max(1, min(24, sample_stride)),
                max_seconds=max(10, min(600, max_seconds)),
                prefer_gpu=prefer_gpu,
            )
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        return jsonify({"candidates": candidates, "count": len(candidates)}), 200
    except Exception as ex:
        return jsonify({"error": f"Face detection failed: {ex}"}), 500


@app.route("/jobs/<job_id>", methods=["GET"])
def get_job_status(job_id):
    job = job_manager.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    response = {
        "job_id": job["job_id"],
        "status": job["status"],
        "phase": job["phase"],
        "progress": job["progress"],
        "message": job.get("message"),
        "error": job.get("error"),
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
        "finished_at": job.get("finished_at"),
    }
    if job.get("status") == "completed":
        response["result"] = job.get("result")

    return jsonify(response)


@app.route("/download-processed/<filename>", methods=["GET"])
def download_processed_video(filename):
    try:
        return send_from_directory(PROCESSED_FOLDER, filename, as_attachment=True)
    except FileNotFoundError:
        return jsonify({"error": "Processed file not found"}), 404


@app.route("/stream-processed/<filename>", methods=["GET"])
def stream_processed_video(filename):
    file_path = os.path.abspath(
    os.path.join(PROCESSED_FOLDER, "..", "..", filename)
)
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
    response = send_file(file_path, mimetype="video/mp4", conditional=True)
    response.headers["Accept-Ranges"] = "bytes"
    return response


@app.route("/list-temp-files", methods=["GET"])
def list_temp_files():
    files = []
    for group, folder in [("uploads", UPLOAD_FOLDER), ("processed", PROCESSED_FOLDER)]:
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if os.path.isdir(path):
                continue
            stat = os.stat(path)
            files.append(
                {
                    "filename": name,
                    "group": group,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                }
            )
    return jsonify({"files": files, "count": len(files)})


@app.route("/cleanup", methods=["POST"])
def cleanup_now():
    before = len(os.listdir(UPLOAD_FOLDER)) + len(os.listdir(PROCESSED_FOLDER))
    cleanup_expired_files([UPLOAD_FOLDER, PROCESSED_FOLDER], TEMP_FILE_EXPIRY_HOURS)
    after = len(os.listdir(UPLOAD_FOLDER)) + len(os.listdir(PROCESSED_FOLDER))
    return jsonify(
        {"message": "Cleanup completed", "deleted": before - after, "remaining": after}
    )


@app.errorhandler(413)
def too_large(_error):
    return jsonify({"error": "File too large. Max size is 200MB."}), 413


@app.errorhandler(404)
def not_found(_error):
    return jsonify({"error": "Endpoint not found"}), 404


if __name__ == "__main__":
    print("Starting Context-Aware Video Censor Backend...")
    print(f"Upload folder: {os.path.abspath(UPLOAD_FOLDER)}")
    print(f"Processed folder: {os.path.abspath(PROCESSED_FOLDER)}")
    print(f"Max file size: {MAX_FILE_SIZE / (1024 * 1024):.0f}MB")
    print("Starting cleanup scheduler...")
    _start_cleanup_scheduler()
    debug_mode = str(os.getenv("FLASK_DEBUG", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    app.run(debug=debug_mode, host="0.0.0.0", port=5000, use_reloader=debug_mode)
