import React, { useState, useRef } from 'react';
import axios from 'axios';

const Home = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processedResult, setProcessedResult] = useState(null);
  const [error, setError] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef(null);

  const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:5000';

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file && file.type.startsWith('video/')) {
      setSelectedFile(file);
      setError(null);
      setProcessedResult(null);
      setUploadProgress(0);
    } else {
      setError('Please select a valid video file');
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = (event) => {
    event.preventDefault();
    setDragOver(false);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setDragOver(false);
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith('video/')) {
      setSelectedFile(file);
      setError(null);
      setProcessedResult(null);
      setUploadProgress(0);
    } else {
      setError('Please drop a valid video file');
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDuration = (seconds) => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  const uploadAndProcessVideo = async () => {
    if (!selectedFile) {
      setError('Please select a video file first');
      return;
    }

    setIsProcessing(true);
    setError(null);
    setUploadProgress(0);

    const formData = new FormData();
    formData.append('video', selectedFile);

    try {
      const response = await axios.post(`${BACKEND_URL}/upload-video`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const progress = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total
          );
          setUploadProgress(progress);
        },
      });

      setProcessedResult(response.data);
      setIsProcessing(false);
    } catch (err) {
      console.error('Upload error:', err);
      setError(
        err.response?.data?.error || 
        'Failed to upload and process video. Make sure the backend server is running.'
      );
      setIsProcessing(false);
    }
  };

  const downloadProcessedVideo = () => {
    if (processedResult && processedResult.processed_video_url) {
      window.open(`${BACKEND_URL}${processedResult.processed_video_url}`, '_blank');
    }
  };

  return (
    <div className="container">
      <h1>Video Upload & Preview System</h1>
      
      {/* Upload Section */}
      <div className="upload-section">
        <h2>Upload Video</h2>
        <div 
          className={`upload-area ${dragOver ? 'dragover' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <div className="upload-icon">📁</div>
          <div className="upload-text">
            {selectedFile ? selectedFile.name : 'Click to select or drag and drop a video file'}
          </div>
          <div className="upload-subtext">
            Supported formats: MP4, AVI, MOV, WMV (Max size: 100MB)
          </div>
          <input
            type="file"
            ref={fileInputRef}
            className="file-input"
            accept="video/*"
            onChange={handleFileSelect}
          />
        </div>

        {selectedFile && (
          <div className="file-info">
            <h3>Selected File Details:</h3>
            <div className="file-detail">Name: {selectedFile.name}</div>
            <div className="file-detail">Size: {formatFileSize(selectedFile.size)}</div>
            <div className="file-detail">Type: {selectedFile.type}</div>
          </div>
        )}

        <button 
          className="btn" 
          onClick={uploadAndProcessVideo}
          disabled={!selectedFile || isProcessing}
          style={{ marginTop: '15px' }}
        >
          {isProcessing ? 'Processing...' : 'Upload & Process Video'}
        </button>
      </div>

      {/* Progress Section */}
      {(isProcessing || uploadProgress > 0) && (
        <div className="processing-section">
          <h2>Processing Video</h2>
          <div className="progress-bar">
            <div 
              className="progress-fill" 
              style={{ width: `${uploadProgress}%` }}
            ></div>
          </div>
          <div>Upload Progress: {uploadProgress}%</div>
          {isProcessing && <div className="loading-spinner"></div>}
          {isProcessing && <div>Processing your video, please wait...</div>}
        </div>
      )}

      {/* Video Preview Section */}
      {selectedFile && (
        <div className="preview-section">
          <h2>Video Preview</h2>
          <video 
            className="video-preview"
            controls
            src={URL.createObjectURL(selectedFile)}
          >
            Your browser does not support the video tag.
          </video>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="error-message">
          <strong>Error:</strong> {error}
        </div>
      )}

      {/* Results Section */}
      {processedResult && (
        <div className="result-section">
          <h2>Processing Complete!</h2>
          <div className="success-message">
            Video uploaded and processed successfully!
          </div>
          
          <div className="file-info">
            <h3>Processing Results:</h3>
            <div className="file-detail">Original filename: {processedResult.original_filename}</div>
            <div className="file-detail">File size: {processedResult.file_size}</div>
            <div className="file-detail">Temp storage path: {processedResult.temp_path}</div>
            <div className="file-detail">Upload time: {new Date(processedResult.upload_time).toLocaleString()}</div>
            {processedResult.video_info && (
              <>
                <div className="file-detail">Duration: {formatDuration(processedResult.video_info.duration)}</div>
                <div className="file-detail">Resolution: {processedResult.video_info.width} x {processedResult.video_info.height}</div>
                <div className="file-detail">Frame rate: {processedResult.video_info.fps} fps</div>
              </>
            )}
          </div>

          {processedResult.processed_video_url && (
            <div style={{ marginTop: '20px' }}>
              <h3>Processed Video:</h3>
              <video 
                className="video-preview"
                controls
                src={`${BACKEND_URL}${processedResult.processed_video_url}`}
              >
                Your browser does not support the video tag.
              </video>
              <button 
                className="btn" 
                onClick={downloadProcessedVideo}
                style={{ marginTop: '10px' }}
              >
                Download Processed Video
              </button>
            </div>
          )}

          {processedResult.preview_data && (
            <div style={{ marginTop: '20px' }}>
              <h3>Processing Preview:</h3>
              <pre style={{ 
                background: '#f8f9fa', 
                padding: '15px', 
                borderRadius: '5px', 
                overflow: 'auto' 
              }}>
                {JSON.stringify(processedResult.preview_data, null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default Home;