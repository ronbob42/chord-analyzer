"""Web-based chord player with real-time scrolling timeline."""

import html as html_mod
import json
import mimetypes
import os
import shutil
import socket
import socketserver
import tempfile
import threading
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler
from pathlib import Path

from .analyzer import ChordEvent

DEFAULT_PORT = 8765

CHORD_COLORS = {
    "maj":  "#4A90D9",
    "min":  "#9B59B6",
    "5":    "#1ABC9C",
    "7":    "#E67E22",
    "maj7": "#2ECC71",
    "min7": "#8E44AD",
    "dim":  "#E74C3C",
    "aug":  "#F39C12",
    "N":    "#555555",
}


def _chord_color(chord_name: str) -> str:
    if chord_name in ("N", "--"):
        return CHORD_COLORS["N"]
    parts = chord_name.split()
    if len(parts) >= 2:
        return CHORD_COLORS.get(parts[1], CHORD_COLORS["maj"])
    return CHORD_COLORS["maj"]


def _build_timeline_data(events: list[ChordEvent], total_duration: float) -> list[dict]:
    """Build a gapless sequence of chord/gap blocks for the timeline."""
    blocks: list[dict] = []
    current_time = 0.0

    for event in events:
        if event.start_time > current_time + 0.05:
            blocks.append({
                "chord": "--",
                "start": round(current_time, 2),
                "end": round(event.start_time, 2),
                "duration": round(event.start_time - current_time, 2),
                "confidence": 0,
                "color": CHORD_COLORS["N"],
            })
        blocks.append({
            "chord": event.chord,
            "start": event.start_time,
            "end": event.end_time,
            "duration": round(event.duration, 2),
            "confidence": event.confidence,
            "color": _chord_color(event.chord),
        })
        current_time = event.end_time

    if current_time < total_duration - 0.05:
        blocks.append({
            "chord": "--",
            "start": round(current_time, 2),
            "end": round(total_duration, 2),
            "duration": round(total_duration - current_time, 2),
            "confidence": 0,
            "color": CHORD_COLORS["N"],
        })

    return blocks


_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Chord Player - {song_title}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: #1a1a2e;
  color: #eee;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
}}
#header {{
  text-align: center;
  padding: 28px 16px 8px;
}}
#header h1 {{
  font-size: 1.3rem;
  font-weight: 500;
  color: #aaa;
}}
#current-chord-display {{
  text-align: center;
  padding: 36px 0 28px;
}}
#current-chord-label {{
  font-size: 5rem;
  font-weight: 700;
  letter-spacing: 2px;
  transition: color 0.12s ease;
  color: #555;
}}
#current-chord-time {{
  font-size: 0.9rem;
  color: #666;
  margin-top: 6px;
}}
#timeline-container {{
  position: relative;
  width: 92%;
  max-width: 1200px;
  height: 80px;
  background: #16213e;
  border-radius: 10px;
  overflow: hidden;
  cursor: pointer;
}}
#timeline {{
  display: flex;
  height: 100%;
  position: absolute;
  top: 0;
  left: 0;
}}
.chord-block {{
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.72rem;
  font-weight: 600;
  color: rgba(255,255,255,0.85);
  border-right: 1px solid rgba(0,0,0,0.25);
  flex-shrink: 0;
  user-select: none;
  overflow: hidden;
  white-space: nowrap;
  transition: filter 0.1s ease;
}}
.chord-block.active {{
  filter: brightness(1.4);
  box-shadow: inset 0 0 16px rgba(255,255,255,0.15);
}}
#playhead {{
  position: absolute;
  top: 0;
  left: 50%;
  width: 2px;
  height: 100%;
  background: #fff;
  z-index: 10;
  pointer-events: none;
  box-shadow: 0 0 8px rgba(255,255,255,0.5);
}}
#controls {{
  display: flex;
  align-items: center;
  gap: 24px;
  padding: 28px 0 16px;
}}
#play-btn {{
  background: #4A90D9;
  color: white;
  border: none;
  border-radius: 28px;
  padding: 12px 36px;
  font-size: 1.05rem;
  cursor: pointer;
  transition: background 0.15s;
  font-weight: 500;
}}
#play-btn:hover {{ background: #5ba0e9; }}
#time-display {{
  font-size: 0.95rem;
  color: #888;
  font-variant-numeric: tabular-nums;
  min-width: 110px;
}}
#progress-bar-container {{
  width: 92%;
  max-width: 1200px;
  height: 6px;
  background: #16213e;
  border-radius: 3px;
  margin-top: 12px;
  cursor: pointer;
  position: relative;
}}
#progress-bar {{
  height: 100%;
  background: #4A90D9;
  border-radius: 3px;
  width: 0%;
  transition: width 0.1s linear;
}}
#hint {{
  color: #555;
  font-size: 0.75rem;
  margin-top: 20px;
}}
@media (max-width: 600px) {{
  #current-chord-label {{ font-size: 3.2rem; }}
  #current-chord-display {{ padding: 20px 0 16px; }}
  #timeline-container {{ height: 60px; width: 96%; }}
  #progress-bar-container {{ width: 96%; height: 10px; }}
  #controls {{ gap: 16px; padding: 20px 0 12px; }}
  #play-btn {{ padding: 14px 32px; font-size: 1.1rem; }}
  #hint {{ display: none; }}
  #header h1 {{ font-size: 1rem; }}
}}
</style>
</head>
<body>

<div id="header">
  <h1>{song_title}</h1>
</div>

<div id="current-chord-display">
  <div id="current-chord-label">--</div>
  <div id="current-chord-time">0:00</div>
</div>

<div id="timeline-container">
  <div id="timeline"></div>
  <div id="playhead"></div>
</div>

<div id="progress-bar-container">
  <div id="progress-bar"></div>
</div>

<div id="controls">
  <button id="play-btn">&#9654;  Play</button>
  <span id="time-display">0:00 / 0:00</span>
</div>

<div id="hint">Space = play/pause &middot; &larr;&rarr; = seek 5s</div>

<audio id="audio" preload="auto" playsinline>
  <source src="{audio_src}" type="{audio_type}">
</audio>

<script>
(function() {{
  var chords = {chord_data_json};
  var audio = document.getElementById('audio');
  var timeline = document.getElementById('timeline');
  var timelineContainer = document.getElementById('timeline-container');
  var playBtn = document.getElementById('play-btn');
  var currentChordLabel = document.getElementById('current-chord-label');
  var currentChordTime = document.getElementById('current-chord-time');
  var timeDisplay = document.getElementById('time-display');
  var progressBar = document.getElementById('progress-bar');
  var progressContainer = document.getElementById('progress-bar-container');

  var PX_PER_SEC = 50;
  var totalDuration = {total_duration};
  var timelineWidth = totalDuration * PX_PER_SEC;
  var containerWidth = timelineContainer.clientWidth;

  timeline.style.width = timelineWidth + 'px';

  // Build chord blocks
  chords.forEach(function(c, i) {{
    var block = document.createElement('div');
    block.className = 'chord-block';
    block.style.width = (c.duration * PX_PER_SEC) + 'px';
    block.style.backgroundColor = c.color;
    block.dataset.index = i;
    if (c.duration * PX_PER_SEC > 32) {{
      block.textContent = c.chord;
    }}
    timeline.appendChild(block);
  }});

  var activeIndex = -1;
  var rafId = null;

  function formatTime(sec) {{
    var m = Math.floor(sec / 60);
    var s = Math.floor(sec % 60);
    return m + ':' + (s < 10 ? '0' : '') + s;
  }}

  function updateDisplay() {{
    var t = audio.currentTime;

    // Scroll timeline: playhead at center, timeline moves left
    var offset = t * PX_PER_SEC;
    var halfContainer = containerWidth / 2;
    var shift = -(offset - halfContainer);
    shift = Math.min(shift, 0);
    shift = Math.max(shift, -(timelineWidth - containerWidth));
    timeline.style.transform = 'translateX(' + shift + 'px)';

    // Progress bar
    progressBar.style.width = (t / totalDuration * 100) + '%';

    // Find current chord (binary search-ish for efficiency)
    var newIndex = -1;
    for (var i = 0; i < chords.length; i++) {{
      if (t >= chords[i].start && t < chords[i].end) {{
        newIndex = i;
        break;
      }}
    }}

    if (newIndex !== activeIndex) {{
      if (activeIndex >= 0 && timeline.children[activeIndex]) {{
        timeline.children[activeIndex].classList.remove('active');
      }}
      activeIndex = newIndex;
      if (activeIndex >= 0) {{
        timeline.children[activeIndex].classList.add('active');
        currentChordLabel.textContent = chords[activeIndex].chord;
        currentChordLabel.style.color = chords[activeIndex].color;
      }} else {{
        currentChordLabel.textContent = '--';
        currentChordLabel.style.color = '#555';
      }}
    }}

    currentChordTime.textContent = formatTime(t);
    timeDisplay.textContent = formatTime(t) + ' / ' + formatTime(totalDuration);
  }}

  function tick() {{
    updateDisplay();
    if (!audio.paused) {{
      rafId = requestAnimationFrame(tick);
    }}
  }}

  audio.addEventListener('play', function() {{
    playBtn.innerHTML = '&#10074;&#10074;  Pause';
    tick();
  }});
  audio.addEventListener('pause', function() {{
    playBtn.innerHTML = '&#9654;  Play';
    if (rafId) cancelAnimationFrame(rafId);
  }});
  audio.addEventListener('ended', function() {{
    playBtn.innerHTML = '&#9654;  Play';
    if (rafId) cancelAnimationFrame(rafId);
    updateDisplay();
  }});

  playBtn.addEventListener('click', function() {{
    if (audio.paused) {{
      var p = audio.play();
      if (p && p.catch) {{
        p.catch(function(err) {{
          console.warn('Play failed:', err);
          // Try loading first, then play
          audio.load();
          audio.addEventListener('canplay', function onReady() {{
            audio.removeEventListener('canplay', onReady);
            audio.play();
          }});
        }});
      }}
    }} else {{
      audio.pause();
    }}
  }});

  // Click to seek on timeline
  timelineContainer.addEventListener('click', function(e) {{
    var rect = timelineContainer.getBoundingClientRect();
    var clickX = e.clientX - rect.left;
    var currentTransform = parseFloat(
      (timeline.style.transform || '').replace('translateX(', '').replace('px)', '')
    ) || 0;
    var timelineX = clickX - currentTransform;
    var seekTime = timelineX / PX_PER_SEC;
    audio.currentTime = Math.max(0, Math.min(seekTime, totalDuration));
    updateDisplay();
  }});

  // Click to seek on progress bar
  progressContainer.addEventListener('click', function(e) {{
    var rect = progressContainer.getBoundingClientRect();
    var ratio = (e.clientX - rect.left) / rect.width;
    audio.currentTime = Math.max(0, Math.min(ratio * totalDuration, totalDuration));
    updateDisplay();
  }});

  // Touch seek on timeline
  timelineContainer.addEventListener('touchstart', function(e) {{
    e.preventDefault();
    var touch = e.touches[0];
    var rect = timelineContainer.getBoundingClientRect();
    var clickX = touch.clientX - rect.left;
    var currentTransform = parseFloat(
      (timeline.style.transform || '').replace('translateX(', '').replace('px)', '')
    ) || 0;
    var timelineX = clickX - currentTransform;
    var seekTime = timelineX / PX_PER_SEC;
    audio.currentTime = Math.max(0, Math.min(seekTime, totalDuration));
    updateDisplay();
  }});

  // Touch seek on progress bar
  progressContainer.addEventListener('touchstart', function(e) {{
    e.preventDefault();
    var touch = e.touches[0];
    var rect = progressContainer.getBoundingClientRect();
    var ratio = (touch.clientX - rect.left) / rect.width;
    audio.currentTime = Math.max(0, Math.min(ratio * totalDuration, totalDuration));
    updateDisplay();
  }});

  // Keyboard shortcuts
  document.addEventListener('keydown', function(e) {{
    if (e.code === 'Space') {{
      e.preventDefault();
      playBtn.click();
    }} else if (e.code === 'ArrowLeft') {{
      audio.currentTime = Math.max(0, audio.currentTime - 5);
      updateDisplay();
    }} else if (e.code === 'ArrowRight') {{
      audio.currentTime = Math.min(totalDuration, audio.currentTime + 5);
      updateDisplay();
    }}
  }});

  // Handle window resize
  window.addEventListener('resize', function() {{
    containerWidth = timelineContainer.clientWidth;
    updateDisplay();
  }});

  updateDisplay();
}})();
</script>
</body>
</html>
"""


def generate_player_html(
    events: list[ChordEvent],
    metadata: dict,
    audio_src: str,
    audio_type: str,
    total_duration: float,
) -> str:
    """Generate self-contained HTML for the chord player.

    Args:
        audio_src: Full audio source — a URL path like '/audio/abc' or a
                   data URI like 'data:audio/mpeg;base64,...'.
        audio_type: MIME type, e.g. 'audio/wav' or 'audio/mpeg'.
    """
    blocks = _build_timeline_data(events, total_duration)
    chord_json = json.dumps(blocks)
    title = html_mod.escape(f"{metadata.get('artist', 'Unknown')} - {metadata.get('title', 'Unknown')}")

    return _HTML_TEMPLATE.format(
        song_title=title,
        audio_src=audio_src,
        audio_type=audio_type,
        chord_data_json=chord_json,
        total_duration=total_duration,
    )


class RangeHTTPRequestHandler(SimpleHTTPRequestHandler):
    """HTTP handler with Range request support for audio seeking."""

    def do_GET(self):
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            super().do_GET()
            return

        range_header = self.headers.get("Range")
        if range_header is None:
            super().do_GET()
            return

        try:
            range_spec = range_header.strip().split("=")[1]
            parts = range_spec.split("-")
            file_size = os.path.getsize(path)
            start = int(parts[0]) if parts[0] else 0
            end = int(parts[1]) if parts[1] else file_size - 1
            end = min(end, file_size - 1)
            length = end - start + 1
        except (ValueError, IndexError):
            self.send_error(416, "Invalid Range")
            return

        content_type, _ = mimetypes.guess_type(path)
        content_type = content_type or "application/octet-stream"

        self.send_response(206)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

        try:
            with open(path, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(65536, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except BrokenPipeError:
            pass  # browser cancelled the request

    def log_message(self, format, *args):
        pass  # suppress request logging


def _find_free_port(start: int = DEFAULT_PORT) -> int:
    for port in range(start, start + 100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    raise RuntimeError("Could not find a free port")


def launch_player(
    events: list[ChordEvent],
    metadata: dict,
    audio_path: Path,
) -> None:
    """Start the web-based chord player.

    Opens a browser with the chord timeline and audio playback.
    Blocks until Ctrl+C, then cleans up.
    """
    import librosa

    total_duration = librosa.get_duration(path=str(audio_path))

    serve_dir = Path(tempfile.mkdtemp(prefix="chord_player_"))
    server = None
    try:
        # Copy audio into serving directory
        audio_dest = serve_dir / audio_path.name
        shutil.copy2(str(audio_path), str(audio_dest))

        # Generate HTML
        html_content = generate_player_html(
            events, metadata, f"/{audio_path.name}", "audio/wav", total_duration
        )
        (serve_dir / "index.html").write_text(html_content, encoding="utf-8")

        # Start server
        port = _find_free_port()
        handler = partial(RangeHTTPRequestHandler, directory=str(serve_dir))
        server = socketserver.TCPServer(("127.0.0.1", port), handler)
        server.allow_reuse_address = True

        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

        url = f"http://localhost:{port}/"
        webbrowser.open(url)

        from rich.console import Console
        console = Console()
        console.print(f"\n  [bold green]Chord player:[/] {url}")
        console.print("  [dim]Press Ctrl+C to stop.[/]\n")

        threading.Event().wait()

    except KeyboardInterrupt:
        pass
    finally:
        if server:
            server.shutdown()
        shutil.rmtree(str(serve_dir), ignore_errors=True)
