import React, { useEffect, useMemo, useRef, useState } from 'react';
import axios from 'axios';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:5000';

const initialOptions = {
    blur_faces: true,
    blur_background_faces: true,
    preserve_primary_subjects: true,
    primary_subject_count: 1,
    blur_sensitive_text: false,
    detect_nudity: true,
    nudity_policy_mode: 'streaming_strict',
    censor_sensitive_audio: true,
    keep_audio: true,
    trusted_face_threshold: 0.82,
    nudity_threshold: 0.82,
    nudity_sample_stride: 5,
    nudity_min_relative_area: 0.01,
    nudity_consecutive_hits: 2,
    nudity_strict_labels: true,
    blur_strength_face: 1.2,
    blur_strength_nudity: 1.55,
    temporal_smoothing_alpha: 0.7,
    temporal_blur_threshold: 0.5,
};

function Home() {
    const fileInputRef = useRef(null);
    const [videoFile, setVideoFile] = useState(null);
    const [candidateFaces, setCandidateFaces] = useState([]);
    const [detectingFaces, setDetectingFaces] = useState(false);
    const [dragOver, setDragOver] = useState(false);

    const [options, setOptions] = useState(initialOptions);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [processing, setProcessing] = useState(false);
    const [job, setJob] = useState(null);
    const [processingProgress, setProcessingProgress] = useState(0);
    const [result, setResult] = useState(null);
    const [error, setError] = useState('');
    const [previewError, setPreviewError] = useState(false);

    const videoPreviewUrl = useMemo(() => {
        if (!videoFile) return '';
        return URL.createObjectURL(videoFile);
    }, [videoFile]);

    const onFileSelected = (file) => {
        if (!file) return;
        if (!file.type.startsWith('video/')) {
            setError('Please upload a valid video file.');
            return;
        }
        setVideoFile(file);
        setResult(null);
        setError('');
        setUploadProgress(0);
        setPreviewError(false);
    };

    const handleDrop = (event) => {
        event.preventDefault();
        setDragOver(false);
        onFileSelected(event.dataTransfer.files?.[0]);
    };

    const setOption = (key, value) => {
        setOptions((prev) => ({ ...prev, [key]: value }));
    };

    const dataUrlToBlob = async (dataUrl) => {
        const response = await fetch(dataUrl);
        return response.blob();
    };

    const detectCandidateFaces = async () => {
        if (!videoFile) {
            setError('Upload a video first.');
            return;
        }

        setDetectingFaces(true);
        setError('');
        try {
            const formData = new FormData();
            formData.append('video', videoFile);
            formData.append('max_candidates', '12');
            formData.append('sample_stride', '6');

            const response = await axios.post(`${BACKEND_URL}/face-candidates`, formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });

            const faces = (response.data?.candidates || []).map((item) => ({
                ...item,
                selected: false,
            }));
            setCandidateFaces(faces);
        } catch (err) {
            setError(err?.response?.data?.error || 'Could not extract face candidates from video.');
        } finally {
            setDetectingFaces(false);
        }
    };

    const toggleCandidateFace = (candidateId) => {
        setCandidateFaces((prev) =>
            prev.map((face) =>
                face.candidate_id === candidateId
                    ? { ...face, selected: !face.selected }
                    : face
            )
        );
    };

    const selectAllCandidates = (selected) => {
        setCandidateFaces((prev) => prev.map((face) => ({ ...face, selected })));
    };

    useEffect(() => {
        if (!job?.id || !processing) return undefined;

        const interval = setInterval(async () => {
            try {
                const response = await axios.get(`${BACKEND_URL}/jobs/${job.id}`);
                const data = response.data;
                setJob((prev) => ({
                    ...(prev || {}),
                    status: data.status,
                    phase: data.phase,
                    message: data.message,
                }));
                setProcessingProgress(Number(data.progress || 0));

                if (data.status === 'completed' && data.result) {
                    setResult(data.result);
                    setPreviewError(false);
                    setProcessing(false);
                    clearInterval(interval);
                }
                if (data.status === 'failed') {
                    setError(data.error || data.message || 'Video processing failed.');
                    setProcessing(false);
                    clearInterval(interval);
                }
            } catch (_err) {
                setError('Failed to query processing status.');
                setProcessing(false);
                clearInterval(interval);
            }
        }, 1400);

        return () => clearInterval(interval);
    }, [job?.id, processing]);

    const processVideo = async () => {
        if (!videoFile) {
            setError('Upload a video first.');
            return;
        }

        setProcessing(true);
        setError('');
        setResult(null);
        setUploadProgress(0);
        setProcessingProgress(0);
        setJob(null);
        setPreviewError(false);

        const formData = new FormData();
        formData.append('video', videoFile);
        formData.append('options', JSON.stringify(options));

        const selectedCandidates = candidateFaces.filter((face) => face.selected);
        for (const face of selectedCandidates) {
            const blob = await dataUrlToBlob(face.image_data_url);
            formData.append('trusted_faces', blob, `${face.candidate_id}.jpg`);
        }

        try {
            const response = await axios.post(`${BACKEND_URL}/process-video`, formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
                onUploadProgress: (event) => {
                    if (!event.total) return;
                    setUploadProgress(Math.round((event.loaded * 100) / event.total));
                },
                timeout: 0,
            });

            if (response.status === 202 && response.data?.job_id) {
                setUploadProgress(100);
                setJob({
                    id: response.data.job_id,
                    status: response.data.status,
                    phase: 'queued',
                    message: response.data.message,
                });
            } else {
                setResult(response.data);
                setProcessing(false);
            }
        } catch (err) {
            setError(
                err?.response?.data?.error ||
                'Processing failed. Ensure backend is running and dependencies are installed.'
            );
            setProcessing(false);
        }
    };

    return (
        <div className="app-shell">
            <div className="ambient-shape ambient-shape-1" />
            <div className="ambient-shape ambient-shape-2" />

            <main className="layout">
                <section className="hero card">
                    <p className="eyebrow">Selective Blur Studio</p>
                    <h1>Blur everything except the faces you approve.</h1>
                    <p className="hero-copy">
                        Upload an MP4, scan face clusters from the video, and export a full-FPS
                        result with explicit content and unapproved faces blurred.
                    </p>
                </section>

                <section className="card grid two-col">
                    <div>
                        <h2>1) Video Input</h2>
                        <div
                            className={`dropzone ${dragOver ? 'dragging' : ''}`}
                            onClick={() => fileInputRef.current?.click()}
                            onDragOver={(e) => {
                                e.preventDefault();
                                setDragOver(true);
                            }}
                            onDragLeave={(e) => {
                                e.preventDefault();
                                setDragOver(false);
                            }}
                            onDrop={handleDrop}
                        >
                            <p className="dropzone-title">
                                {videoFile ? videoFile.name : 'Drop video here or click to choose'}
                            </p>
                            <p className="dropzone-sub">MP4 recommended. Full-FPS output preserved.</p>
                            <input
                                ref={fileInputRef}
                                className="hidden"
                                type="file"
                                accept="video/*"
                                onChange={(e) => onFileSelected(e.target.files?.[0])}
                            />
                        </div>
                        {videoFile && (
                            <div className="pill-row">
                                <span className="pill">{(videoFile.size / (1024 * 1024)).toFixed(2)} MB</span>
                                <span className="pill">{videoFile.type || 'video/*'}</span>
                            </div>
                        )}

                        {videoPreviewUrl && (
                            <video className="video" controls src={videoPreviewUrl} />
                        )}
                    </div>

                    <div>
                        <h2>2) Allowed Faces (from video)</h2>
                        <p className="help-text">
                            Scan the video for face clusters, then click the faces you want to keep.
                            Everything else will be blurred by default.
                        </p>
                        <div className="candidate-actions">
                            <button className="ghost-btn" onClick={detectCandidateFaces} disabled={!videoFile || detectingFaces}>
                                {detectingFaces ? 'Scanning Frames...' : 'Scan Face Clusters'}
                            </button>
                            <button className="ghost-btn" onClick={() => selectAllCandidates(true)} disabled={candidateFaces.length === 0}>
                                Keep All
                            </button>
                            <button className="ghost-btn" onClick={() => selectAllCandidates(false)} disabled={candidateFaces.length === 0}>
                                Keep None
                            </button>
                        </div>

                        {candidateFaces.length > 0 && (
                            <div className="candidate-grid">
                                {candidateFaces.map((face, index) => (
                                    <button
                                        type="button"
                                        key={face.candidate_id}
                                        className={`candidate-card ${face.selected ? 'selected' : ''}`}
                                        onClick={() => toggleCandidateFace(face.candidate_id)}
                                    >
                                        <img src={face.image_data_url} alt={face.candidate_id} />
                                        <span>Cluster {index + 1} · {face.frames_seen} frames</span>
                                    </button>
                                ))}
                            </div>
                        )}
                    </div>
                </section>

                <section className="card">
                    <h2>3) Censor Policy</h2>
                    <div className="options-grid">
                        <label className="toggle">
                            <input
                                type="checkbox"
                                checked={options.blur_faces}
                                onChange={(e) => setOption('blur_faces', e.target.checked)}
                            />
                            <span>Blur all faces except allowed clusters</span>
                        </label>

                        <label className="toggle">
                            <input
                                type="checkbox"
                                checked={options.blur_background_faces}
                                onChange={(e) => setOption('blur_background_faces', e.target.checked)}
                            />
                            <span>Aggressive blur for small/far faces</span>
                        </label>

                        <label className="toggle">
                            <input
                                type="checkbox"
                                checked={options.preserve_primary_subjects}
                                onChange={(e) => setOption('preserve_primary_subjects', e.target.checked)}
                            />
                            <span>Fallback: auto-keep top faces if none selected</span>
                        </label>

                        <label className="toggle">
                            <input
                                type="checkbox"
                                checked={options.blur_sensitive_text}
                                onChange={(e) => setOption('blur_sensitive_text', e.target.checked)}
                            />
                            <span>Blur text overlays/subtitles (optional)</span>
                        </label>

                        <label className="toggle">
                            <input
                                type="checkbox"
                                checked={options.detect_nudity}
                                onChange={(e) => setOption('detect_nudity', e.target.checked)}
                            />
                            <span>Detect and blur explicit content (strict)</span>
                        </label>

                        <label className="toggle">
                            <input
                                type="checkbox"
                                checked={options.censor_sensitive_audio}
                                onChange={(e) => setOption('censor_sensitive_audio', e.target.checked)}
                            />
                            <span>Mute audio during explicit events</span>
                        </label>

                        <label className="toggle">
                            <input
                                type="checkbox"
                                checked={options.keep_audio}
                                onChange={(e) => setOption('keep_audio', e.target.checked)}
                            />
                            <span>Keep original audio track (passthrough)</span>
                        </label>
                    </div>

                    <div className="sliders">
                        <label>
                            Auto-keep faces (fallback): <strong>{options.primary_subject_count}</strong>
                            <input
                                type="range"
                                min="1"
                                max="5"
                                value={options.primary_subject_count}
                                onChange={(e) => setOption('primary_subject_count', Number(e.target.value))}
                            />
                        </label>

                        <label>
                            Face match threshold: <strong>{options.trusted_face_threshold.toFixed(2)}</strong>
                            <input
                                type="range"
                                min="0.5"
                                max="0.95"
                                step="0.01"
                                value={options.trusted_face_threshold}
                                onChange={(e) => setOption('trusted_face_threshold', Number(e.target.value))}
                            />
                        </label>

                        <label>
                            Explicit confidence threshold: <strong>{options.nudity_threshold.toFixed(2)}</strong>
                            <input
                                type="range"
                                min="0.3"
                                max="0.95"
                                step="0.01"
                                value={options.nudity_threshold}
                                onChange={(e) => setOption('nudity_threshold', Number(e.target.value))}
                            />
                        </label>

                        <label>
                            Explicit policy mode
                            <select
                                value={options.nudity_policy_mode}
                                onChange={(e) => setOption('nudity_policy_mode', e.target.value)}
                            >
                                <option value="porn_only">Strictest (adult-only)</option>
                                <option value="balanced">Balanced</option>
                                <option value="streaming_strict">Streaming-Strict (recommended)</option>
                            </select>
                        </label>

                        <label>
                            Temporal smoothing alpha: <strong>{options.temporal_smoothing_alpha.toFixed(2)}</strong>
                            <input
                                type="range"
                                min="0.05"
                                max="0.95"
                                step="0.01"
                                value={options.temporal_smoothing_alpha}
                                onChange={(e) => setOption('temporal_smoothing_alpha', Number(e.target.value))}
                            />
                        </label>

                        <label>
                            Temporal blur threshold: <strong>{options.temporal_blur_threshold.toFixed(2)}</strong>
                            <input
                                type="range"
                                min="0.2"
                                max="0.95"
                                step="0.01"
                                value={options.temporal_blur_threshold}
                                onChange={(e) => setOption('temporal_blur_threshold', Number(e.target.value))}
                            />
                        </label>
                    </div>

                    <button className="primary-btn" disabled={processing || !videoFile} onClick={processVideo}>
                        {processing ? 'Processing Video...' : 'Run Context-Aware Censoring'}
                    </button>

                    {processing && (
                        <div className="progress-wrap">
                            <p className="muted">Upload progress</p>
                            <div className="progress-track">
                                <div className="progress-fill" style={{ width: `${uploadProgress}%` }} />
                            </div>
                            <p>{uploadProgress}% uploaded.</p>

                            <p className="muted">Processing progress</p>
                            <div className="progress-track">
                                <div className="progress-fill" style={{ width: `${processingProgress}%` }} />
                            </div>
                            <p>
                                {processingProgress.toFixed(1)}% {job?.phase ? `(${job.phase})` : ''}
                            </p>
                            {job?.message && <p className="muted">{job.message}</p>}
                        </div>
                    )}

                    {error && <div className="error-box">{error}</div>}
                </section>

                {result && (
                    <section className="card result">
                        <h2>4) Result</h2>
                        <p className="success">{result.message}</p>

                        <video
                            className="video"
                            controls
                            src={`${BACKEND_URL}${result.processed_video_url}`}
                            onError={() => setPreviewError(true)}
                        />

                        {previewError && (
                            <div className="error-box">
                                Processed video could not be previewed in the browser. Download and open it locally, or ensure FFmpeg is available and reprocess to force web-compatible H.264 output.
                            </div>
                        )}

                        <div className="action-row">
                            <a className="primary-btn as-link" href={`${BACKEND_URL}${result.download_video_url}`}>
                                Download Processed Video
                            </a>
                        </div>

                        <div className="stats-grid">
                            <div className="stat"><span>Faces blurred</span><strong>{result.report?.faces_blurred ?? 0}</strong></div>
                            <div className="stat"><span>Faces preserved</span><strong>{result.report?.faces_preserved ?? 0}</strong></div>
                            <div className="stat"><span>Text regions blurred</span><strong>{result.report?.text_regions_blurred ?? 0}</strong></div>
                            <div className="stat"><span>Explicit regions blurred</span><strong>{result.report?.nudity_regions_blurred ?? 0}</strong></div>
                            <div className="stat"><span>Trusted refs used</span><strong>{result.report?.trusted_reference_faces ?? 0}</strong></div>
                            <div className="stat"><span>Audio mute intervals</span><strong>{result.report?.audio_censored_intervals?.length ?? 0}</strong></div>
                        </div>

                        <details>
                            <summary>Detailed Report JSON</summary>
                            <pre>{JSON.stringify(result.report, null, 2)}</pre>
                        </details>
                    </section>
                )}
            </main>
        </div>
    );
}

export default Home;
