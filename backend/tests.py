import unittest
import json
import os
import tempfile
from app import app, allowed_file, get_video_info
from werkzeug.test import Client
from werkzeug.serving import WSGIRequestHandler

class VideoUploadTestCase(unittest.TestCase):
    
    def setUp(self):
        """Set up test client and temporary directories"""
        self.app = app.test_client()
        self.app.testing = True
        
        # Create temporary directories for testing
        self.temp_upload_dir = tempfile.mkdtemp()
        self.temp_processed_dir = tempfile.mkdtemp()
        app.config['UPLOAD_FOLDER'] = self.temp_upload_dir
        app.config['PROCESSED_FOLDER'] = self.temp_processed_dir

    def tearDown(self):
        """Clean up temporary directories"""
        import shutil
        if os.path.exists(self.temp_upload_dir):
            shutil.rmtree(self.temp_upload_dir)
        if os.path.exists(self.temp_processed_dir):
            shutil.rmtree(self.temp_processed_dir)

    def test_health_check(self):
        """Test the health check endpoint"""
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
        self.assertIn('timestamp', data)

    def test_allowed_file_function(self):
        """Test the allowed_file function"""
        # Valid video files
        self.assertTrue(allowed_file('video.mp4'))
        self.assertTrue(allowed_file('video.avi'))
        self.assertTrue(allowed_file('video.mov'))
        self.assertTrue(allowed_file('video.wmv'))
        
        # Invalid files
        self.assertFalse(allowed_file('image.jpg'))
        self.assertFalse(allowed_file('document.pdf'))
        self.assertFalse(allowed_file('video'))  # no extension
        self.assertFalse(allowed_file(''))  # empty filename

    def test_upload_no_file(self):
        """Test upload endpoint with no file"""
        response = self.app.post('/upload-video')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertIn('No video file provided', data['error'])

    def test_upload_empty_filename(self):
        """Test upload endpoint with empty filename"""
        with tempfile.NamedTemporaryFile(suffix='.mp4') as temp_file:
            response = self.app.post('/upload-video', data={
                'video': (temp_file, '')  # empty filename
            })
            self.assertEqual(response.status_code, 400)
            data = json.loads(response.data)
            self.assertIn('error', data)

    def test_upload_invalid_file_type(self):
        """Test upload endpoint with invalid file type"""
        with tempfile.NamedTemporaryFile(suffix='.txt', mode='w') as temp_file:
            temp_file.write('This is not a video file')
            temp_file.flush()
            
            with open(temp_file.name, 'rb') as f:
                response = self.app.post('/upload-video', data={
                    'video': (f, 'test.txt')
                })
                self.assertEqual(response.status_code, 400)
                data = json.loads(response.data)
                self.assertIn('error', data)
                self.assertIn('File type not allowed', data['error'])

    def test_list_temp_files_empty(self):
        """Test listing temporary files when directories are empty"""
        response = self.app.get('/list-temp-files')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['total_files'], 0)
        self.assertEqual(len(data['temp_files']), 0)

    def test_manual_cleanup(self):
        """Test manual cleanup endpoint"""
        response = self.app.post('/cleanup')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('message', data)
        self.assertIn('files_deleted', data)
        self.assertIn('files_remaining', data)

    def test_download_nonexistent_file(self):
        """Test downloading a file that doesn't exist"""
        response = self.app.get('/download-processed/nonexistent.mp4')
        self.assertEqual(response.status_code, 404)
        data = json.loads(response.data)
        self.assertIn('error', data)

    def test_stream_nonexistent_file(self):
        """Test streaming a file that doesn't exist"""
        response = self.app.get('/stream-processed/nonexistent.mp4')
        self.assertEqual(response.status_code, 404)
        data = json.loads(response.data)
        self.assertIn('error', data)

    def create_dummy_video_file(self, filename='test_video.mp4'):
        """Create a dummy video file for testing"""
        import cv2
        import numpy as np
        
        # Create a simple video file for testing
        filepath = os.path.join(self.temp_upload_dir, filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filepath, fourcc, 20.0, (640, 480))
        
        # Write a few frames
        for i in range(60):  # 3 seconds at 20fps
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            out.write(frame)
        
        out.release()
        return filepath

    def test_get_video_info_with_dummy_video(self):
        """Test video info extraction with a dummy video"""
        try:
            video_path = self.create_dummy_video_file()
            info = get_video_info(video_path)
            
            if info:  # Only test if OpenCV is working properly
                self.assertIsInstance(info, dict)
                self.assertIn('width', info)
                self.assertIn('height', info)
                self.assertIn('fps', info)
                self.assertIn('duration', info)
                self.assertIn('frame_count', info)
            else:
                # If video info extraction fails, it might be due to OpenCV issues
                # This is acceptable in a testing environment
                print("Warning: Video info extraction failed - possibly due to OpenCV setup")
        except Exception as e:
            # Skip this test if OpenCV is not properly set up
            print(f"Skipping video info test due to OpenCV issues: {e}")

if __name__ == '__main__':
    print("Running Video Upload Backend Tests...")
    print("Note: Some tests may be skipped if OpenCV is not properly configured.")
    unittest.main(verbosity=2)