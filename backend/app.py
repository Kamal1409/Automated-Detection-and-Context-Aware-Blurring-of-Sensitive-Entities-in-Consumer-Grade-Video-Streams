from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import uuid
import shutil
import cv2
import json
from datetime import datetime, timedelta
import mimetypes
import threading
import time

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

# Configuration
UPLOAD_FOLDER = 'temp_storage'
PROCESSED_FOLDER = 'processed_videos'
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'wmv', 'flv', 'webm', 'mkv'}
TEMP_FILE_EXPIRY_HOURS = 24  # Files will be deleted after 24 hours

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_video_info(video_path):
    """Extract video information using OpenCV"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        return {
            'width': width,
            'height': height,
            'fps': round(fps, 2),
            'duration': round(duration, 2),
            'frame_count': int(frame_count)
        }
    except Exception as e:
        print(f"Error extracting video info: {str(e)}")
        return None

def process_video_preview(video_path, output_path):
    """
    Process video to generate a preview (example: extract first 10 seconds)
    This is where you would implement your specific video processing logic
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer for preview (first 10 seconds)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        max_frames = int(fps * 10)  # 10 seconds preview
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Add processing effects here (example: add timestamp)
            timestamp = f"Frame: {frame_count}"
            cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            out.write(frame)
            frame_count += 1
        
        cap.release()
        out.release()
        return True
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return False

def cleanup_temp_files():
    """Clean up temporary files older than TEMP_FILE_EXPIRY_HOURS"""
    try:
        current_time = datetime.now()
        for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER]:
            for filename in os.listdir(folder):
                filepath = os.path.join(folder, filename)
                if os.path.isfile(filepath):
                    file_modified = datetime.fromtimestamp(os.path.getmtime(filepath))
                    if current_time - file_modified > timedelta(hours=TEMP_FILE_EXPIRY_HOURS):
                        os.remove(filepath)
                        print(f"Deleted expired file: {filepath}")
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")

def start_cleanup_scheduler():
    """Start a background thread for periodic cleanup"""
    def cleanup_scheduler():
        while True:
            time.sleep(3600)  # Run every hour
            cleanup_temp_files()
    
    cleanup_thread = threading.Thread(target=cleanup_scheduler, daemon=True)
    cleanup_thread.start()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/upload-video', methods=['POST'])
def upload_video():
    """Handle video upload and processing"""
    try:
        # Check if a file was uploaded
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed. Please upload a video file.'}), 400
        
        # Generate unique filename
        original_filename = secure_filename(file.filename)
        file_extension = original_filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save the uploaded file
        file.save(temp_filepath)
        file_size = os.path.getsize(temp_filepath)
        
        # Get video information
        video_info = get_video_info(temp_filepath)
        
        # Process the video (generate preview)
        processed_filename = f"processed_{unique_filename}"
        processed_filepath = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
        
        processing_success = process_video_preview(temp_filepath, processed_filepath)
        
        # Prepare response data
        response_data = {
            'message': 'Video uploaded and processed successfully',
            'original_filename': original_filename,
            'temp_filename': unique_filename,
            'temp_path': temp_filepath,
            'file_size': f"{file_size / (1024*1024):.2f} MB",
            'upload_time': datetime.now().isoformat(),
            'video_info': video_info,
            'preview_data': {
                'processing_applied': 'Frame numbering and 10-second preview',
                'original_duration': video_info['duration'] if video_info else 'Unknown',
                'preview_duration': min(10, video_info['duration']) if video_info else 'Unknown'
            }
        }
        
        # Add processed video URL if processing was successful
        if processing_success:
            response_data['processed_video_url'] = f'/download-processed/{processed_filename}'
        else:
            response_data['processing_error'] = 'Failed to process video, but original uploaded successfully'
        
        return jsonify(response_data), 200
    
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/download-processed/<filename>')
def download_processed_video(filename):
    """Download processed video file"""
    try:
        return send_from_directory(app.config['PROCESSED_FOLDER'], filename, as_attachment=True)
    except FileNotFoundError:
        return jsonify({'error': 'Processed file not found'}), 404

@app.route('/stream-processed/<filename>')
def stream_processed_video(filename):
    """Stream processed video file for preview"""
    try:
        file_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        # Set appropriate MIME type
        mimetype = mimetypes.guess_type(file_path)[0] or 'video/mp4'
        return send_file(file_path, mimetype=mimetype)
    except Exception as e:
        return jsonify({'error': f'Error streaming file: {str(e)}'}), 500

@app.route('/list-temp-files', methods=['GET'])
def list_temp_files():
    """List all files in temporary storage"""
    try:
        files = []
        for folder_name, folder_path in [('uploads', UPLOAD_FOLDER), ('processed', PROCESSED_FOLDER)]:
            for filename in os.listdir(folder_path):
                filepath = os.path.join(folder_path, filename)
                if os.path.isfile(filepath):
                    stat = os.stat(filepath)
                    files.append({
                        'filename': filename,
                        'folder': folder_name,
                        'size': f"{stat.st_size / (1024*1024):.2f} MB",
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        'age_hours': round((datetime.now() - datetime.fromtimestamp(stat.st_mtime)).total_seconds() / 3600, 2)
                    })
        
        return jsonify({
            'temp_files': files,
            'total_files': len(files),
            'cleanup_interval_hours': TEMP_FILE_EXPIRY_HOURS
        }), 200
    except Exception as e:
        return jsonify({'error': f'Error listing files: {str(e)}'}), 500

@app.route('/cleanup', methods=['POST'])
def manual_cleanup():
    """Manually trigger cleanup of expired files"""
    try:
        files_before = len(os.listdir(UPLOAD_FOLDER)) + len(os.listdir(PROCESSED_FOLDER))
        cleanup_temp_files()
        files_after = len(os.listdir(UPLOAD_FOLDER)) + len(os.listdir(PROCESSED_FOLDER))
        files_deleted = files_before - files_after
        
        return jsonify({
            'message': 'Cleanup completed',
            'files_deleted': files_deleted,
            'files_remaining': files_after
        }), 200
    except Exception as e:
        return jsonify({'error': f'Cleanup failed: {str(e)}'}), 500

@app.errorhandler(413)
def file_too_large(error):
    return jsonify({'error': 'File too large. Maximum size is 100MB'}), 413

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Starting Video Upload Backend Server...")
    print(f"Upload folder: {os.path.abspath(UPLOAD_FOLDER)}")
    print(f"Processed folder: {os.path.abspath(PROCESSED_FOLDER)}")
    print(f"Max file size: {MAX_FILE_SIZE / (1024*1024):.0f}MB")
    print(f"Supported formats: {', '.join(ALLOWED_EXTENSIONS)}")
    print("Starting cleanup scheduler...")
    
    # Start cleanup scheduler
    start_cleanup_scheduler()
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)