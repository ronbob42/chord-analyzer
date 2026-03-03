"""Flask web app for the Spotify Chord Analyzer."""

import json
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path

import librosa
from flask import Flask, Response, jsonify, request, send_file

from chord_analyzer.analyzer import (
    analyze_audio, _extract_chroma, _match_chords, _stabilize_labels,
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

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac", ".wma", ".opus"}
MAX_UPLOAD_MB = 50


def _cleanup_loop():
    """Background thread that removes expired sessions."""
    while True:
        time.sleep(120)
        now = time.time()
        expired = []
        with _sessions_lock:
            for sid, s in _sessions.items():
                if now - s["created_at"] > SESSION_TTL:
                    expired.append(sid)
            for sid in expired:
                s = _sessions.pop(sid)
                shutil.rmtree(s["tmpdir"], ignore_errors=True)


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

  <p class="hint">Analysis takes 10-60 seconds depending on song length.</p>
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

// Submit handlers
uploadBtn.addEventListener('click', function() {
  if (!selectedFile) return;
  var formData = new FormData();
  formData.append('file', selectedFile);
  submitUploadSSE(formData, uploadBtn);
});

document.getElementById('spotify-btn').addEventListener('click', function() {
  var url = document.getElementById('url').value.trim();
  if (!url) return;
  var btn = document.getElementById('spotify-btn');
  submitSpotifySSE(url, btn);
});

// Also submit Spotify URL on Enter key
document.getElementById('url').addEventListener('keydown', function(e) {
  if (e.key === 'Enter') {
    e.preventDefault();
    document.getElementById('spotify-btn').click();
  }
});

function handleSSEMessage(data, btn) {
  var spinner = document.getElementById('spinner');
  var status = document.getElementById('status');
  var error = document.getElementById('error');

  if (data.step === 'done') {
    if (data.is_preview) {
      document.getElementById('preview-note').style.display = 'block';
    }
    status.textContent = 'Ready!';
    window.location.href = '/play/' + data.session_id;
  } else if (data.step === 'error') {
    error.textContent = data.msg || 'Something went wrong.';
    spinner.classList.remove('active');
    btn.disabled = false;
  } else {
    status.textContent = data.msg || data.step;
  }
}

function submitSpotifySSE(url, btn) {
  var spinner = document.getElementById('spinner');
  var status = document.getElementById('status');
  var error = document.getElementById('error');

  btn.disabled = true;
  error.textContent = '';
  status.textContent = 'Starting...';
  spinner.classList.add('active');

  var es = new EventSource('/api/analyze/stream?url=' + encodeURIComponent(url));
  es.onmessage = function(e) {
    var data = JSON.parse(e.data);
    handleSSEMessage(data, btn);
    if (data.step === 'done' || data.step === 'error') {
      es.close();
    }
  };
  es.onerror = function() {
    es.close();
    error.textContent = 'Connection lost. Please try again.';
    spinner.classList.remove('active');
    btn.disabled = false;
  };
}

function submitUploadSSE(formData, btn) {
  var spinner = document.getElementById('spinner');
  var status = document.getElementById('status');
  var error = document.getElementById('error');

  btn.disabled = true;
  error.textContent = '';
  status.textContent = 'Uploading...';
  spinner.classList.add('active');

  fetch('/api/upload/stream', {method: 'POST', body: formData}).then(function(resp) {
    var reader = resp.body.getReader();
    var decoder = new TextDecoder();
    var buffer = '';

    function read() {
      reader.read().then(function(result) {
        if (result.done) return;
        buffer += decoder.decode(result.value, {stream: true});
        var lines = buffer.split('\\n');
        buffer = lines.pop();
        lines.forEach(function(line) {
          if (line.startsWith('data: ')) {
            var data = JSON.parse(line.slice(6));
            handleSSEMessage(data, btn);
          }
        });
        read();
      }).catch(function() {
        error.textContent = 'Connection lost. Please try again.';
        spinner.classList.remove('active');
        btn.disabled = false;
      });
    }
    read();
  }).catch(function() {
    error.textContent = 'Network error. Please try again.';
    spinner.classList.remove('active');
    btn.disabled = false;
  });
}
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# API & routes
# ---------------------------------------------------------------------------


def _analyze_and_store(audio_path: str, metadata: dict, tmpdir: str) -> dict:
    """Shared analysis logic for both upload and Spotify endpoints."""
    events = analyze_audio(audio_path)
    if not events:
        shutil.rmtree(tmpdir, ignore_errors=True)
        return {"error": "No chords detected in this track.", "status": 422}

    duration = librosa.get_duration(path=str(audio_path))
    session_id = uuid.uuid4().hex[:12]

    html = generate_player_html(
        events,
        metadata,
        audio_src=f"/audio/{session_id}",
        audio_type="audio/wav",
        total_duration=duration,
    )

    with _sessions_lock:
        _sessions[session_id] = {
            "audio_path": str(audio_path),
            "html": html,
            "tmpdir": tmpdir,
            "created_at": time.time(),
        }

    return {"session_id": session_id, "is_preview": metadata.get("is_preview", False)}


@app.route("/")
def index():
    return LANDING_HTML


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    body = request.get_json(silent=True) or {}
    url = (body.get("url") or "").strip()

    if not url or "spotify.com/track/" not in url:
        return jsonify(error="Please enter a valid Spotify track URL."), 400

    tmpdir = tempfile.mkdtemp(prefix="chord_web_")

    try:
        audio_path, metadata = download_track(url, tmpdir)
        result = _analyze_and_store(str(audio_path), metadata, tmpdir)
        if "error" in result:
            return jsonify(error=result["error"]), result["status"]
        return jsonify(result)

    except (ValueError, DownloadError) as e:
        shutil.rmtree(tmpdir, ignore_errors=True)
        return jsonify(error=str(e)), 400
    except Exception as e:
        shutil.rmtree(tmpdir, ignore_errors=True)
        return jsonify(error=f"Analysis failed: {e}"), 500


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files:
        return jsonify(error="No file uploaded."), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify(error="No file selected."), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify(error=f"Unsupported format ({ext}). Use MP3, WAV, OGG, FLAC, or M4A."), 400

    tmpdir = tempfile.mkdtemp(prefix="chord_upload_")

    try:
        # Save uploaded file
        safe_name = re.sub(r'[^\w\s\-.]', '', file.filename).strip()[:100]
        saved_path = Path(tmpdir) / safe_name
        file.save(str(saved_path))

        if saved_path.stat().st_size > MAX_UPLOAD_MB * 1024 * 1024:
            shutil.rmtree(tmpdir, ignore_errors=True)
            return jsonify(error=f"File too large (max {MAX_UPLOAD_MB} MB)."), 400

        # Convert to WAV if needed (librosa handles most formats, but WAV is safest)
        if ext != ".wav":
            wav_path = Path(tmpdir) / (saved_path.stem + ".wav")
            import subprocess
            try:
                subprocess.run(
                    ["ffmpeg", "-i", str(saved_path), "-ar", "22050", "-ac", "1", "-y", str(wav_path)],
                    capture_output=True, timeout=120,
                )
            except (subprocess.TimeoutExpired, FileNotFoundError):
                # Fall back to letting librosa handle it directly
                wav_path = saved_path

            if wav_path.exists() and wav_path.stat().st_size > 0:
                audio_path = wav_path
            else:
                audio_path = saved_path
        else:
            audio_path = saved_path

        # Extract title from filename
        title = saved_path.stem.replace("_", " ").replace("-", " ")
        metadata = {"artist": "Uploaded", "title": title, "is_preview": False}

        result = _analyze_and_store(str(audio_path), metadata, tmpdir)
        if "error" in result:
            return jsonify(error=result["error"]), result["status"]
        return jsonify(result)

    except Exception as e:
        shutil.rmtree(tmpdir, ignore_errors=True)
        return jsonify(error=f"Analysis failed: {e}"), 500


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


def _sse_event(step: str, **kwargs) -> str:
    """Format a server-sent event line."""
    payload = {"step": step, "msg": PROGRESS_MESSAGES.get(step, step), **kwargs}
    return f"data: {json.dumps(payload)}\n\n"


@app.route("/api/analyze/stream")
def api_analyze_stream():
    url = (request.args.get("url") or "").strip()
    if not url or "spotify.com/track/" not in url:
        def error_gen():
            yield _sse_event("error", msg="Please enter a valid Spotify track URL.")
        return Response(error_gen(), content_type="text/event-stream")

    def generate():
        tmpdir = tempfile.mkdtemp(prefix="chord_web_")
        try:
            yield _sse_event("fetching")

            yield _sse_event("downloading")
            audio_path, metadata = download_track(url, tmpdir)

            yield _sse_event("loading")
            y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE)
            if len(y) / sr < 5:
                yield _sse_event("error", msg="Audio file is too short for chord analysis.")
                return

            yield _sse_event("extracting")
            chroma = _extract_chroma(y, sr)

            yield _sse_event("matching")
            labels, confidences = _match_chords(chroma, CONFIDENCE_THRESHOLD)
            labels, confidences = _stabilize_labels(labels, confidences)
            events = _merge_events(labels, confidences, sr, HOP_LENGTH, MIN_CHORD_DURATION)

            if not events:
                yield _sse_event("error", msg="No chords detected in this track.")
                shutil.rmtree(tmpdir, ignore_errors=True)
                return

            yield _sse_event("building")
            duration = len(y) / sr
            session_id = uuid.uuid4().hex[:12]
            html = generate_player_html(
                events, metadata,
                audio_src=f"/audio/{session_id}",
                audio_type="audio/wav",
                total_duration=duration,
            )
            with _sessions_lock:
                _sessions[session_id] = {
                    "audio_path": str(audio_path),
                    "html": html,
                    "tmpdir": tmpdir,
                    "created_at": time.time(),
                }
            yield _sse_event("done", session_id=session_id, is_preview=metadata.get("is_preview", False))

        except (ValueError, DownloadError) as e:
            shutil.rmtree(tmpdir, ignore_errors=True)
            yield _sse_event("error", msg=str(e))
        except Exception as e:
            shutil.rmtree(tmpdir, ignore_errors=True)
            yield _sse_event("error", msg=f"Analysis failed: {e}")

    return Response(generate(), content_type="text/event-stream")


@app.route("/api/upload/stream", methods=["POST"])
def api_upload_stream():
    if "file" not in request.files:
        def error_gen():
            yield _sse_event("error", msg="No file uploaded.")
        return Response(error_gen(), content_type="text/event-stream")

    file = request.files["file"]
    if not file.filename:
        def error_gen():
            yield _sse_event("error", msg="No file selected.")
        return Response(error_gen(), content_type="text/event-stream")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        def error_gen():
            yield _sse_event("error", msg=f"Unsupported format ({ext}). Use MP3, WAV, OGG, FLAC, or M4A.")
        return Response(error_gen(), content_type="text/event-stream")

    def generate():
        tmpdir = tempfile.mkdtemp(prefix="chord_upload_")
        try:
            yield _sse_event("saving")
            safe_name = re.sub(r'[^\w\s\-.]', '', file.filename).strip()[:100]
            saved_path = Path(tmpdir) / safe_name
            file.save(str(saved_path))

            if saved_path.stat().st_size > MAX_UPLOAD_MB * 1024 * 1024:
                shutil.rmtree(tmpdir, ignore_errors=True)
                yield _sse_event("error", msg=f"File too large (max {MAX_UPLOAD_MB} MB).")
                return

            if ext != ".wav":
                yield _sse_event("converting")
                wav_path = Path(tmpdir) / (saved_path.stem + ".wav")
                import subprocess
                try:
                    subprocess.run(
                        ["ffmpeg", "-i", str(saved_path), "-ar", "22050", "-ac", "1", "-y", str(wav_path)],
                        capture_output=True, timeout=120,
                    )
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    wav_path = saved_path
                audio_path = wav_path if (wav_path.exists() and wav_path.stat().st_size > 0) else saved_path
            else:
                audio_path = saved_path

            yield _sse_event("loading")
            y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE)
            if len(y) / sr < 5:
                yield _sse_event("error", msg="Audio file is too short for chord analysis.")
                return

            yield _sse_event("extracting")
            chroma = _extract_chroma(y, sr)

            yield _sse_event("matching")
            labels, confidences = _match_chords(chroma, CONFIDENCE_THRESHOLD)
            labels, confidences = _stabilize_labels(labels, confidences)
            events = _merge_events(labels, confidences, sr, HOP_LENGTH, MIN_CHORD_DURATION)

            if not events:
                shutil.rmtree(tmpdir, ignore_errors=True)
                yield _sse_event("error", msg="No chords detected in this track.")
                return

            yield _sse_event("building")
            duration = len(y) / sr
            session_id = uuid.uuid4().hex[:12]
            title = saved_path.stem.replace("_", " ").replace("-", " ")
            metadata = {"artist": "Uploaded", "title": title, "is_preview": False}
            html = generate_player_html(
                events, metadata,
                audio_src=f"/audio/{session_id}",
                audio_type="audio/wav",
                total_duration=duration,
            )
            with _sessions_lock:
                _sessions[session_id] = {
                    "audio_path": str(audio_path),
                    "html": html,
                    "tmpdir": tmpdir,
                    "created_at": time.time(),
                }
            yield _sse_event("done", session_id=session_id, is_preview=False)

        except Exception as e:
            shutil.rmtree(tmpdir, ignore_errors=True)
            yield _sse_event("error", msg=f"Analysis failed: {e}")

    return Response(generate(), content_type="text/event-stream")


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

            resp = Response(data, 206, mimetype="audio/wav")
            resp.headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
            resp.headers["Accept-Ranges"] = "bytes"
            resp.headers["Content-Length"] = length
            return resp

    return send_file(audio_path, mimetype="audio/wav")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
