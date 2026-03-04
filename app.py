"""Flask web app for the Spotify Chord Analyzer."""

import gc
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

import librosa
from flask import Flask, Response, jsonify, request, send_file

from chord_analyzer.analyzer import (
    _extract_chroma, _match_chords, _stabilize_labels,
    _merge_events, HOP_LENGTH, CONFIDENCE_THRESHOLD, MIN_CHORD_DURATION,
    SAMPLE_RATE,
)
from chord_analyzer.downloader import DownloadError, download_track
from chord_analyzer.player import generate_player_html

app = Flask(__name__)

# Session storage: session_id -> dict with paths and results
_sessions: dict[str, dict] = {}
_sessions_lock = threading.Lock()
SESSION_TTL = 1800  # 30 minutes

# Job storage: job_id -> dict with status info
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
JOB_TTL = 600  # 10 minutes

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac", ".wma", ".opus"}
MAX_UPLOAD_MB = 50

MIME_MAP = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".ogg": "audio/ogg",
            ".m4a": "audio/mp4", ".flac": "audio/flac", ".opus": "audio/opus"}

PROGRESS_MESSAGES = {
    "fetching": "Fetching track info from Spotify...",
    "downloading": "Downloading audio...",
    "saving": "Saving uploaded file...",
    "converting": "Converting to WAV...",
    "loading": "Loading audio waveform...",
    "extracting": "Extracting harmonic content...",
    "matching": "Matching chord templates...",
    "building": "Building player...",
}


def _audio_mime(path: str) -> str:
    return MIME_MAP.get(Path(path).suffix.lower(), "audio/mpeg")


def _set_job_step(job_id: str, step: str, **kwargs):
    """Update a job's current step."""
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id]["step"] = step
            _jobs[job_id]["msg"] = PROGRESS_MESSAGES.get(step, step)
            _jobs[job_id].update(kwargs)


def _cleanup_loop():
    """Background thread that removes expired sessions and jobs."""
    while True:
        time.sleep(120)
        now = time.time()
        # Clean sessions
        expired = []
        with _sessions_lock:
            for sid, s in _sessions.items():
                if now - s["created_at"] > SESSION_TTL:
                    expired.append(sid)
            for sid in expired:
                s = _sessions.pop(sid)
                shutil.rmtree(s["tmpdir"], ignore_errors=True)
        # Clean jobs
        expired_jobs = []
        with _jobs_lock:
            for jid, j in _jobs.items():
                if now - j["created_at"] > JOB_TTL:
                    expired_jobs.append(jid)
            for jid in expired_jobs:
                _jobs.pop(jid)


threading.Thread(target=_cleanup_loop, daemon=True).start()

# ---------------------------------------------------------------------------
# Landing page
# ---------------------------------------------------------------------------

LANDING_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Chord Analyzer</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: #1a1a2e;
  color: #eee;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}
.container {
  text-align: center;
  max-width: 640px;
  width: 90%;
}
h1 {
  font-size: 2.4rem;
  margin-bottom: 8px;
  color: #4A90D9;
}
.subtitle {
  color: #888;
  margin-bottom: 36px;
  font-size: 1.05rem;
}

/* Tabs */
.tabs {
  display: flex;
  justify-content: center;
  gap: 0;
  margin-bottom: 28px;
}
.tab {
  padding: 10px 28px;
  font-size: 0.95rem;
  font-weight: 600;
  cursor: pointer;
  background: #16213e;
  color: #888;
  border: 2px solid #2a2a4a;
  transition: all 0.2s;
}
.tab:first-child { border-radius: 10px 0 0 10px; }
.tab:last-child  { border-radius: 0 10px 10px 0; }
.tab.active {
  background: #4A90D9;
  color: white;
  border-color: #4A90D9;
}

/* Panels */
.panel { display: none; }
.panel.active { display: block; }

/* Spotify input */
.input-group {
  display: flex;
  gap: 12px;
  margin-bottom: 12px;
}
input[type="text"] {
  flex: 1;
  padding: 14px 18px;
  border-radius: 10px;
  border: 2px solid #2a2a4a;
  background: #16213e;
  color: #eee;
  font-size: 1rem;
  outline: none;
  transition: border-color 0.2s;
}
input[type="text"]:focus { border-color: #4A90D9; }
input[type="text"]::placeholder { color: #555; }

/* Upload drop zone */
.drop-zone {
  border: 2px dashed #2a2a4a;
  border-radius: 12px;
  padding: 40px 20px;
  background: #16213e;
  cursor: pointer;
  transition: border-color 0.2s, background 0.2s;
  margin-bottom: 12px;
}
.drop-zone:hover, .drop-zone.dragover {
  border-color: #4A90D9;
  background: #1c2a4a;
}
.drop-zone p {
  color: #888;
  font-size: 0.95rem;
}
.drop-zone .icon {
  font-size: 2rem;
  margin-bottom: 8px;
  color: #4A90D9;
}
.drop-zone .filename {
  color: #4A90D9;
  font-weight: 600;
  margin-top: 8px;
}
input[type="file"] { display: none; }

/* Shared */
.submit-btn {
  padding: 14px 28px;
  border-radius: 10px;
  border: none;
  background: #4A90D9;
  color: white;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.15s;
  white-space: nowrap;
}
.submit-btn:hover { background: #5ba0e9; }
.submit-btn:disabled { background: #333; cursor: not-allowed; }
.error {
  color: #e74c3c;
  margin-top: 12px;
  font-size: 0.9rem;
  min-height: 24px;
}
.preview-note {
  color: #f39c12;
  font-size: 0.85rem;
  margin-top: 8px;
  display: none;
}
.spinner {
  display: none;
  margin: 24px auto 0;
  text-align: center;
}
.spinner.active { display: block; }
.spinner .dots span {
  display: inline-block;
  width: 10px; height: 10px;
  background: #4A90D9;
  border-radius: 50%;
  margin: 0 4px;
  animation: bounce 1.4s infinite ease-in-out both;
}
.spinner .dots span:nth-child(1) { animation-delay: -0.32s; }
.spinner .dots span:nth-child(2) { animation-delay: -0.16s; }
@keyframes bounce {
  0%, 80%, 100% { transform: scale(0); }
  40% { transform: scale(1); }
}
.spinner .status {
  color: #888;
  margin-top: 12px;
  font-size: 0.9rem;
}
.hint {
  color: #555;
  font-size: 0.8rem;
  margin-top: 32px;
}
</style>
</head>
<body>
<div class="container">
  <h1>Chord Analyzer</h1>
  <p class="subtitle">Analyze chords in any song with a real-time player</p>

  <div class="tabs">
    <div class="tab active" data-panel="upload-panel">Upload File</div>
    <div class="tab" data-panel="spotify-panel">Spotify URL</div>
  </div>

  <!-- Upload panel -->
  <div id="upload-panel" class="panel active">
    <div class="drop-zone" id="drop-zone">
      <div class="icon">&#127925;</div>
      <p>Drop an audio file here or click to browse</p>
      <p style="color:#555;font-size:0.8rem;margin-top:6px">MP3, WAV, OGG, FLAC, M4A (max 50 MB)</p>
      <div class="filename" id="file-name"></div>
    </div>
    <input type="file" id="file-input" accept=".mp3,.wav,.ogg,.flac,.m4a,.aac,.wma,.opus">
    <button class="submit-btn" id="upload-btn" disabled>Analyze</button>
  </div>

  <!-- Spotify panel -->
  <div id="spotify-panel" class="panel">
    <div class="input-group">
      <input type="text" id="url" placeholder="https://open.spotify.com/track/..." autocomplete="off">
      <button class="submit-btn" id="spotify-btn">Analyze</button>
    </div>
    <div class="preview-note" id="preview-note">
      Note: Using a 30-second preview. Full song was unavailable for this track.
    </div>
  </div>

  <div class="error" id="error"></div>

  <div class="spinner" id="spinner">
    <div class="dots"><span></span><span></span><span></span></div>
    <div class="status" id="status">Analyzing chords...</div>
  </div>

  <p class="hint">Analysis takes 1-3 minutes depending on song length.</p>
</div>

<script>
// Tab switching
document.querySelectorAll('.tab').forEach(function(tab) {
  tab.addEventListener('click', function() {
    document.querySelectorAll('.tab').forEach(function(t) { t.classList.remove('active'); });
    document.querySelectorAll('.panel').forEach(function(p) { p.classList.remove('active'); });
    tab.classList.add('active');
    document.getElementById(tab.dataset.panel).classList.add('active');
    document.getElementById('error').textContent = '';
  });
});

// File upload
var dropZone = document.getElementById('drop-zone');
var fileInput = document.getElementById('file-input');
var uploadBtn = document.getElementById('upload-btn');
var fileName = document.getElementById('file-name');
var selectedFile = null;

dropZone.addEventListener('click', function() { fileInput.click(); });
dropZone.addEventListener('dragover', function(e) { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', function() { dropZone.classList.remove('dragover'); });
dropZone.addEventListener('drop', function(e) {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  if (e.dataTransfer.files.length > 0) selectFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', function() {
  if (fileInput.files.length > 0) selectFile(fileInput.files[0]);
});

function selectFile(file) {
  selectedFile = file;
  fileName.textContent = file.name + ' (' + (file.size / 1048576).toFixed(1) + ' MB)';
  uploadBtn.disabled = false;
}

// Poll for job status
function pollJob(jobId, btn) {
  var spinner = document.getElementById('spinner');
  var status = document.getElementById('status');
  var error = document.getElementById('error');
  var pollErrors = 0;

  var interval = setInterval(function() {
    fetch('/api/job/' + jobId).then(function(resp) {
      return resp.json();
    }).then(function(data) {
      pollErrors = 0;
      if (data.step === 'done') {
        clearInterval(interval);
        status.textContent = 'Ready!';
        if (data.is_preview) {
          document.getElementById('preview-note').style.display = 'block';
        }
        window.location.href = '/play/' + data.session_id;
      } else if (data.step === 'error') {
        clearInterval(interval);
        error.textContent = data.msg || 'Something went wrong.';
        spinner.classList.remove('active');
        btn.disabled = false;
      } else {
        status.textContent = data.msg || data.step;
      }
    }).catch(function() {
      pollErrors++;
      if (pollErrors > 10) {
        clearInterval(interval);
        error.textContent = 'Lost connection to server. Please try again.';
        spinner.classList.remove('active');
        btn.disabled = false;
      }
    });
  }, 3000);
}

// Spotify submit
document.getElementById('spotify-btn').addEventListener('click', function() {
  var url = document.getElementById('url').value.trim();
  if (!url) return;
  var btn = document.getElementById('spotify-btn');
  var spinner = document.getElementById('spinner');
  var status = document.getElementById('status');
  var error = document.getElementById('error');

  btn.disabled = true;
  error.textContent = '';
  status.textContent = 'Starting...';
  spinner.classList.add('active');

  fetch('/api/analyze/start', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({url: url})
  }).then(function(resp) {
    return resp.json();
  }).then(function(data) {
    if (data.error) {
      error.textContent = data.error;
      spinner.classList.remove('active');
      btn.disabled = false;
    } else {
      status.textContent = 'Downloading audio...';
      pollJob(data.job_id, btn);
    }
  }).catch(function() {
    error.textContent = 'Network error. Please try again.';
    spinner.classList.remove('active');
    btn.disabled = false;
  });
});

// Also submit Spotify URL on Enter key
document.getElementById('url').addEventListener('keydown', function(e) {
  if (e.key === 'Enter') {
    e.preventDefault();
    document.getElementById('spotify-btn').click();
  }
});

// Upload submit
uploadBtn.addEventListener('click', function() {
  if (!selectedFile) return;
  var btn = uploadBtn;
  var spinner = document.getElementById('spinner');
  var status = document.getElementById('status');
  var error = document.getElementById('error');

  btn.disabled = true;
  error.textContent = '';
  status.textContent = 'Uploading...';
  spinner.classList.add('active');

  var formData = new FormData();
  formData.append('file', selectedFile);

  fetch('/api/upload/start', {
    method: 'POST',
    body: formData
  }).then(function(resp) {
    return resp.json();
  }).then(function(data) {
    if (data.error) {
      error.textContent = data.error;
      spinner.classList.remove('active');
      btn.disabled = false;
    } else {
      status.textContent = 'Processing...';
      pollJob(data.job_id, btn);
    }
  }).catch(function() {
    error.textContent = 'Network error. Please try again.';
    spinner.classList.remove('active');
    btn.disabled = false;
  });
});
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# API & routes
# ---------------------------------------------------------------------------


def _run_analysis(job_id: str, audio_path: str, metadata: dict, tmpdir: str):
    """Run the full analysis pipeline in a background thread."""
    try:
        _set_job_step(job_id, "loading")

        # Pre-convert to low-rate mono WAV with ffmpeg (C-native, low memory)
        # so librosa never decodes the full-quality audio in Python.
        analysis_wav = Path(tmpdir) / "_analysis_lr.wav"
        try:
            subprocess.run(
                ["ffmpeg", "-i", str(audio_path),
                 "-ar", str(SAMPLE_RATE), "-ac", "1", "-t", "120",
                 "-y", str(analysis_wav)],
                capture_output=True, timeout=120,
            )
            load_path = str(analysis_wav) if (analysis_wav.exists() and analysis_wav.stat().st_size > 0) else str(audio_path)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            load_path = str(audio_path)

        y, sr = librosa.load(load_path, sr=SAMPLE_RATE, duration=120)

        # Remove temp analysis WAV immediately to free disk
        if analysis_wav.exists():
            analysis_wav.unlink(missing_ok=True)

        if len(y) / sr < 5:
            _set_job_step(job_id, "error", msg="Audio file is too short for chord analysis.")
            return

        _set_job_step(job_id, "extracting")
        chroma = _extract_chroma(y, sr)
        duration = len(y) / sr  # save before freeing
        del y                   # free raw audio
        gc.collect()

        _set_job_step(job_id, "matching")
        labels, confidences = _match_chords(chroma, CONFIDENCE_THRESHOLD)
        labels, confidences = _stabilize_labels(labels, confidences)
        events = _merge_events(labels, confidences, sr, HOP_LENGTH, MIN_CHORD_DURATION)

        if not events:
            shutil.rmtree(tmpdir, ignore_errors=True)
            _set_job_step(job_id, "error", msg="No chords detected in this track.")
            return

        _set_job_step(job_id, "building")
        session_id = uuid.uuid4().hex[:12]
        html = generate_player_html(
            events, metadata,
            audio_src=f"/audio/{session_id}",
            audio_type=_audio_mime(str(audio_path)),
            total_duration=duration,
        )
        with _sessions_lock:
            _sessions[session_id] = {
                "audio_path": str(audio_path),
                "html": html,
                "tmpdir": tmpdir,
                "created_at": time.time(),
            }
        _set_job_step(job_id, "done",
                      session_id=session_id,
                      is_preview=metadata.get("is_preview", False))

    except Exception as e:
        shutil.rmtree(tmpdir, ignore_errors=True)
        _set_job_step(job_id, "error", msg=f"Analysis failed: {e}")


def _spotify_worker(job_id: str, url: str):
    """Background worker for Spotify URL analysis."""
    tmpdir = tempfile.mkdtemp(prefix="chord_web_")
    try:
        _set_job_step(job_id, "downloading")
        audio_path, metadata = download_track(url, tmpdir)
        _run_analysis(job_id, str(audio_path), metadata, tmpdir)
    except (ValueError, DownloadError) as e:
        shutil.rmtree(tmpdir, ignore_errors=True)
        _set_job_step(job_id, "error", msg=str(e))
    except Exception as e:
        shutil.rmtree(tmpdir, ignore_errors=True)
        _set_job_step(job_id, "error", msg=f"Download failed: {e}")


def _upload_worker(job_id: str, file_bytes: bytes, file_name: str, ext: str):
    """Background worker for file upload analysis."""
    tmpdir = tempfile.mkdtemp(prefix="chord_upload_")
    try:
        _set_job_step(job_id, "saving")
        safe_name = re.sub(r'[^\w\s\-.]', '', file_name).strip()[:100]
        saved_path = Path(tmpdir) / safe_name
        saved_path.write_bytes(file_bytes)

        if saved_path.stat().st_size > MAX_UPLOAD_MB * 1024 * 1024:
            shutil.rmtree(tmpdir, ignore_errors=True)
            _set_job_step(job_id, "error", msg=f"File too large (max {MAX_UPLOAD_MB} MB).")
            return

        if ext != ".wav":
            _set_job_step(job_id, "converting")
            wav_path = Path(tmpdir) / (saved_path.stem + ".wav")
            try:
                subprocess.run(
                    ["ffmpeg", "-i", str(saved_path), "-ar", str(SAMPLE_RATE), "-ac", "1", "-y", str(wav_path)],
                    capture_output=True, timeout=120,
                )
            except (subprocess.TimeoutExpired, FileNotFoundError):
                wav_path = saved_path
            audio_path = wav_path if (wav_path.exists() and wav_path.stat().st_size > 0) else saved_path
        else:
            audio_path = saved_path

        title = saved_path.stem.replace("_", " ").replace("-", " ")
        metadata = {"artist": "Uploaded", "title": title, "is_preview": False}
        _run_analysis(job_id, str(audio_path), metadata, tmpdir)

    except Exception as e:
        shutil.rmtree(tmpdir, ignore_errors=True)
        _set_job_step(job_id, "error", msg=f"Processing failed: {e}")


@app.route("/")
def index():
    return LANDING_HTML


@app.route("/api/analyze/start", methods=["POST"])
def api_analyze_start():
    """Start a Spotify analysis job. Returns job_id immediately."""
    body = request.get_json(silent=True) or {}
    url = (body.get("url") or "").strip()

    if not url or "spotify.com/track/" not in url:
        return jsonify(error="Please enter a valid Spotify track URL."), 400

    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = {
            "step": "fetching",
            "msg": PROGRESS_MESSAGES["fetching"],
            "created_at": time.time(),
        }

    t = threading.Thread(target=_spotify_worker, args=(job_id, url), daemon=True)
    t.start()

    return jsonify(job_id=job_id)


@app.route("/api/upload/start", methods=["POST"])
def api_upload_start():
    """Start a file upload analysis job. Returns job_id immediately."""
    if "file" not in request.files:
        return jsonify(error="No file uploaded."), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify(error="No file selected."), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify(error=f"Unsupported format ({ext}). Use MP3, WAV, OGG, FLAC, or M4A."), 400

    # Read file into memory before returning (request context ends after response)
    file_bytes = file.read()
    file_name = file.filename

    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = {
            "step": "saving",
            "msg": PROGRESS_MESSAGES["saving"],
            "created_at": time.time(),
        }

    t = threading.Thread(target=_upload_worker, args=(job_id, file_bytes, file_name, ext), daemon=True)
    t.start()

    return jsonify(job_id=job_id)


@app.route("/api/job/<job_id>")
def api_job_status(job_id):
    """Poll for job progress. Returns current step and message."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify(step="error", msg="Job not found or expired."), 404
    # Return a copy without internal fields
    return jsonify(
        step=job["step"],
        msg=job.get("msg", ""),
        session_id=job.get("session_id"),
        is_preview=job.get("is_preview", False),
    )


@app.route("/play/<session_id>")
def play(session_id):
    with _sessions_lock:
        session = _sessions.get(session_id)
    if not session:
        return "Session expired or not found.", 404
    return session["html"]


@app.route("/audio/<session_id>")
def audio(session_id):
    with _sessions_lock:
        session = _sessions.get(session_id)
    if not session:
        return "Not found", 404

    audio_path = session["audio_path"]
    file_size = os.path.getsize(audio_path)

    mime = _audio_mime(audio_path)

    range_header = request.headers.get("Range")
    if range_header:
        m = re.match(r"bytes=(\d+)-(\d*)", range_header)
        if m:
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else file_size - 1
            end = min(end, file_size - 1)
            length = end - start + 1

            with open(audio_path, "rb") as f:
                f.seek(start)
                data = f.read(length)

            resp = Response(data, 206, mimetype=mime)
            resp.headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
            resp.headers["Accept-Ranges"] = "bytes"
            resp.headers["Content-Length"] = length
            return resp

    return send_file(audio_path, mimetype=mime)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
