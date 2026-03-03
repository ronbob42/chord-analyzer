"""Download audio from a Spotify URL.

Uses Spotify's embed page to get track metadata (no API key needed).
Audio sources tried in order:
  1. YouTube via yt-dlp (full song)
  2. Spotify 30-second preview clip (last resort)
"""

import json
import logging
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import requests

log = logging.getLogger(__name__)


class DownloadError(Exception):
    pass


SPOTIFY_URL_PATTERN = re.compile(
    r'https?://open\.spotify\.com/track/([a-zA-Z0-9]+)'
)


def _find_bin(name: str) -> str | None:
    """Find an executable, preferring the one in our venv."""
    venv_path = str(Path(sys.executable).parent)
    path = shutil.which(name, path=venv_path)
    if path is None:
        path = shutil.which(name)
    return path


def _get_track_info(spotify_url: str) -> dict:
    """Extract track metadata from Spotify's embed page (no API key needed).

    Returns dict with keys: title, artist, track_id, preview_url (may be None).
    """
    url = spotify_url.strip()
    m = SPOTIFY_URL_PATTERN.search(url)
    if not m:
        raise ValueError(
            "Invalid Spotify URL. Expected format: "
            "https://open.spotify.com/track/<id>"
        )
    track_id = m.group(1)

    try:
        resp = requests.get(
            f"https://open.spotify.com/embed/track/{track_id}",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
        resp.raise_for_status()

        data_match = re.search(
            r'<script id="__NEXT_DATA__" type="application/json">([^<]+)</script>',
            resp.text,
        )
        if not data_match:
            raise DownloadError("Could not parse track data from Spotify embed page.")

        data = json.loads(data_match.group(1))
        entity = data["props"]["pageProps"]["state"]["data"]["entity"]

        title = entity.get("name", "Unknown")
        artists = [a["name"] for a in entity.get("artists", [])]
        artist = ", ".join(artists) if artists else "Unknown"

        # Extract preview URL if available
        preview_url = None
        audio_preview = entity.get("audioPreview")
        if isinstance(audio_preview, dict):
            preview_url = audio_preview.get("url")
        elif isinstance(audio_preview, str):
            preview_url = audio_preview

        return {
            "title": title,
            "artist": artist,
            "track_id": track_id,
            "preview_url": preview_url,
        }

    except (requests.RequestException, KeyError, json.JSONDecodeError) as e:
        raise DownloadError(f"Could not fetch track info from Spotify: {e}")


def _download_spotify_preview(preview_url: str, output_path: Path, safe_name: str) -> Path | None:
    """Download Spotify's 30-second MP3 preview and convert to WAV.

    Returns the WAV path on success, None on failure.
    """
    try:
        resp = requests.get(preview_url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException:
        return None

    mp3_path = output_path / f"{safe_name}_preview.mp3"
    mp3_path.write_bytes(resp.content)

    wav_path = output_path / f"{safe_name}.wav"
    ffmpeg_bin = _find_bin("ffmpeg") or "ffmpeg"
    try:
        subprocess.run(
            [ffmpeg_bin, "-i", str(mp3_path), "-y", str(wav_path)],
            capture_output=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None

    if wav_path.exists() and wav_path.stat().st_size > 0:
        mp3_path.unlink(missing_ok=True)
        return wav_path
    return None


def _find_audio_file(output_path: Path) -> Path | None:
    """Find any audio file in the given directory."""
    for ext in ("*.mp3", "*.wav", "*.m4a", "*.opus", "*.ogg"):
        files = list(output_path.glob(ext))
        if files:
            return files[0]
    return None


def _download_via_ytdlp(search_query: str, output_path: Path, safe_name: str) -> Path | None:
    """Download full song from YouTube via yt-dlp (as Python library).

    Tries multiple YouTube player clients to work around bot detection
    on datacenter IPs. Falls back to SoundCloud if YouTube is blocked.

    Returns the audio file path on success, None on failure.
    """
    try:
        import yt_dlp
    except ImportError:
        return None

    ffmpeg_bin = _find_bin("ffmpeg") or "ffmpeg"
    output_template = str(output_path / f"{safe_name}.%(ext)s")

    base_opts = {
        "format": "bestaudio/best",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "128",
        }],
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "outtmpl": output_template,
        "ffmpeg_location": str(Path(ffmpeg_bin).parent),
        "socket_timeout": 30,
    }

    # Try YouTube with different player clients to bypass bot detection
    player_clients = [
        "default",
        "web_creator",
        "android_vr",
        "mediaconnect",
    ]

    for client in player_clients:
        # Clean up any partial downloads
        for f in output_path.glob(f"{safe_name}.*"):
            f.unlink(missing_ok=True)

        opts = dict(base_opts)
        if client != "default":
            opts["extractor_args"] = {"youtube": {"player_client": [client]}}

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([f"ytsearch1:{search_query}"])
            result = _find_audio_file(output_path)
            if result:
                log.info("YouTube download succeeded with client=%s", client)
                return result
        except Exception as e:
            log.debug("yt-dlp YouTube (client=%s) failed: %s", client, str(e)[:200])
            continue

    # YouTube blocked — try SoundCloud as fallback
    log.info("YouTube blocked, trying SoundCloud...")
    sc_client_id = _get_soundcloud_client_id()
    for f in output_path.glob(f"{safe_name}.*"):
        f.unlink(missing_ok=True)

    sc_opts = dict(base_opts)
    if sc_client_id:
        sc_opts["extractor_args"] = {"soundcloud": {"client_id": [sc_client_id]}}

    try:
        with yt_dlp.YoutubeDL(sc_opts) as ydl:
            ydl.download([f"scsearch1:{search_query}"])
        result = _find_audio_file(output_path)
        if result:
            log.info("SoundCloud download succeeded")
            return result
    except Exception as e:
        log.debug("yt-dlp SoundCloud failed: %s", str(e)[:200])

    return None


def _get_soundcloud_client_id() -> str | None:
    """Extract a fresh SoundCloud client_id from their JS bundles."""
    try:
        resp = requests.get(
            "https://soundcloud.com",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        # Find JS bundle URLs
        js_urls = re.findall(r'https://a-v2\.sndcdn\.com/assets/[^\s"]+\.js', resp.text)
        for js_url in reversed(js_urls):  # Check last bundles first
            try:
                js_resp = requests.get(js_url, timeout=10)
                m = re.search(r'client_id:"([a-zA-Z0-9]+)"', js_resp.text)
                if m:
                    client_id = m.group(1)
                    log.info("Got SoundCloud client_id: %s", client_id[:8] + "...")
                    return client_id
            except Exception:
                continue
    except Exception as e:
        log.debug("Failed to get SoundCloud client_id: %s", str(e)[:200])
    return None


def download_track(spotify_url: str, output_dir: str | None = None) -> tuple[Path, dict]:
    """Download audio from a Spotify URL as WAV.

    Tries YouTube (full song) first, falls back to Spotify's 30-second
    preview clip if YouTube is unavailable.

    Args:
        spotify_url: A Spotify track URL.
        output_dir: Directory to save the file. Uses a temp dir if None.

    Returns:
        Tuple of (path_to_wav, metadata_dict).
        metadata includes 'artist', 'title', and 'is_preview' (bool).

    Raises:
        DownloadError: If all download methods fail.
        ValueError: If the URL is invalid.
    """
    info = _get_track_info(spotify_url)
    artist = info["artist"]
    title = info["title"]
    search_query = f"{artist} - {title}"

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="chord_analyzer_")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    safe_name = re.sub(r'[^\w\s\-]', '', search_query).strip()[:100]
    metadata = {"artist": artist, "title": title}

    # Try 1: full song via YouTube
    wav_path = _download_via_ytdlp(search_query, output_path, safe_name)
    if wav_path:
        metadata["is_preview"] = False
        return wav_path, metadata

    # Try 2: Spotify 30-second preview (last resort)
    if info.get("preview_url"):
        wav_path = _download_spotify_preview(info["preview_url"], output_path, safe_name)
        if wav_path:
            metadata["is_preview"] = True
            return wav_path, metadata

    raise DownloadError(
        "Could not download audio. YouTube may be unavailable and no "
        "Spotify preview exists for this track. Try uploading an audio file instead."
    )
