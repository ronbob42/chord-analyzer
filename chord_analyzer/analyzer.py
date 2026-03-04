"""Core chord detection: audio file -> list of ChordEvent."""

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
from scipy.ndimage import median_filter

from .templates import generate_all_templates

# Tuning parameters
SAMPLE_RATE = 11025  # low rate saves memory; plenty for chord detection
HOP_LENGTH = 2048          # ~186ms per frame at 11025 Hz
HARMONIC_MARGIN = 3.0      # HPSS separation strength
SMOOTH_KERNEL = 9          # median filter width (frames)
CONFIDENCE_THRESHOLD = 0.4  # min cosine similarity to accept a chord
MIN_CHORD_DURATION = 0.3   # seconds


@dataclass
class ChordEvent:
    chord: str
    start_time: float
    end_time: float
    confidence: float

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


def _extract_chroma(y: np.ndarray, sr: int) -> np.ndarray:
    """Extract smoothed, normalized CQT chromagram from harmonic signal."""
    import gc as _gc

    # Harmonic-percussive separation — manual for memory control
    # Use smaller n_fft to reduce STFT memory footprint
    S = librosa.stft(y, n_fft=1024)
    H, _ = librosa.decompose.hpss(S, margin=HARMONIC_MARGIN)
    del S, _
    _gc.collect()
    y_harmonic = librosa.istft(H, length=len(y))
    del H
    _gc.collect()

    # CQT chromagram: (12, T)
    chroma = librosa.feature.chroma_cqt(
        y=y_harmonic,
        sr=sr,
        hop_length=HOP_LENGTH,
        n_chroma=12,
        bins_per_octave=36,
    )
    del y_harmonic
    _gc.collect()

    # Temporal smoothing with median filter
    chroma = median_filter(chroma, size=(1, SMOOTH_KERNEL))

    # L2 normalize each frame
    norms = np.linalg.norm(chroma, axis=0, keepdims=True)
    norms[norms == 0] = 1
    chroma = chroma / norms
    return chroma


def _get_root(chord_name: str) -> str | None:
    """Extract root note from a chord label like 'E min7' -> 'E'."""
    if chord_name == 'N':
        return None
    parts = chord_name.split()
    return parts[0] if parts else None


def _match_chords(chroma: np.ndarray, confidence_threshold: float) -> tuple[list[str], list[float]]:
    """Match each chroma frame to the best chord template via cosine similarity.

    Returns (labels, confidences) — parallel lists, one entry per frame.
    """
    templates = generate_all_templates()
    template_names = list(templates.keys())
    template_matrix = np.array([templates[n] for n in template_names])

    # Normalize templates
    t_norms = np.linalg.norm(template_matrix, axis=1, keepdims=True)
    t_norms[t_norms == 0] = 1
    template_norm = template_matrix / t_norms

    # Cosine similarity: (N_templates, 12) @ (12, T) -> (N_templates, T)
    similarity = template_norm @ chroma

    best_indices = np.argmax(similarity, axis=0)
    max_sims = similarity[best_indices, np.arange(similarity.shape[1])]

    labels = []
    confidences = []
    for i, idx in enumerate(best_indices):
        if max_sims[i] < confidence_threshold:
            labels.append('N')
            confidences.append(float(max_sims[i]))
        else:
            labels.append(template_names[idx])
            confidences.append(float(max_sims[i]))

    return labels, confidences


def _stabilize_labels(labels: list[str], confidences: list[float]) -> tuple[list[str], list[float]]:
    """Stabilize chord quality within root-stable segments.

    Groups consecutive frames that share the same root note, then replaces
    every frame in the group with the most common (majority) chord label.
    This eliminates quality flickering (e.g. C maj -> C 7 -> C maj7 -> C 5)
    when the root is clearly the same throughout.
    """
    from collections import Counter

    if not labels:
        return labels, confidences

    stabilized = list(labels)
    stab_conf = list(confidences)

    seg_start = 0
    seg_root = _get_root(labels[0])

    for i in range(1, len(labels) + 1):
        cur_root = _get_root(labels[i]) if i < len(labels) else None

        if cur_root != seg_root or i == len(labels):
            # End of a root-stable segment: [seg_start, i)
            seg_len = i - seg_start

            if seg_root is not None and seg_len > 1:
                # Find the most common full chord label in this segment
                seg_labels = labels[seg_start:i]
                seg_confs = confidences[seg_start:i]
                counter = Counter(seg_labels)
                winner = counter.most_common(1)[0][0]

                # Compute average confidence for the winning label
                winner_confs = [c for l, c in zip(seg_labels, seg_confs) if l == winner]
                avg_conf = sum(winner_confs) / len(winner_confs)

                for j in range(seg_start, i):
                    stabilized[j] = winner
                    stab_conf[j] = avg_conf

            seg_start = i
            seg_root = cur_root

    return stabilized, stab_conf


def _merge_events(
    labels: list[str],
    confidences: list[float],
    sr: int,
    hop_length: int,
    min_duration: float,
) -> list[ChordEvent]:
    """Merge consecutive identical labels into ChordEvent spans."""
    if not labels:
        return []

    events: list[ChordEvent] = []
    frame_duration = hop_length / sr

    current_label = labels[0]
    current_conf_sum = confidences[0]
    current_count = 1
    start_frame = 0

    for i in range(1, len(labels)):
        if labels[i] == current_label:
            current_conf_sum += confidences[i]
            current_count += 1
        else:
            start_t = start_frame * frame_duration
            end_t = i * frame_duration
            avg_conf = current_conf_sum / current_count
            if current_label != 'N' and (end_t - start_t) >= min_duration:
                events.append(ChordEvent(
                    chord=current_label,
                    start_time=round(start_t, 2),
                    end_time=round(end_t, 2),
                    confidence=round(avg_conf, 3),
                ))
            current_label = labels[i]
            current_conf_sum = confidences[i]
            current_count = 1
            start_frame = i

    # Final segment
    start_t = start_frame * frame_duration
    end_t = len(labels) * frame_duration
    avg_conf = current_conf_sum / current_count
    if current_label != 'N' and (end_t - start_t) >= min_duration:
        events.append(ChordEvent(
            chord=current_label,
            start_time=round(start_t, 2),
            end_time=round(end_t, 2),
            confidence=round(avg_conf, 3),
        ))

    return events


def analyze_audio(
    audio_path: str | Path,
    min_duration: float = MIN_CHORD_DURATION,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    progress_cb=None,
) -> list[ChordEvent]:
    """Full chord analysis pipeline.

    Args:
        audio_path: Path to a WAV or MP3 file.
        min_duration: Minimum chord event duration in seconds.
        confidence_threshold: Minimum cosine similarity to accept a chord.
        progress_cb: Optional callback called with a step name string
            at each stage ("loading", "extracting", "matching").

    Returns:
        List of ChordEvent sorted by start_time.
    """
    if progress_cb:
        progress_cb("loading")
    y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE)
    if len(y) / sr < 5:
        raise ValueError("Audio file is too short for meaningful chord analysis.")

    if progress_cb:
        progress_cb("extracting")
    chroma = _extract_chroma(y, sr)

    if progress_cb:
        progress_cb("matching")
    labels, confidences = _match_chords(chroma, confidence_threshold)
    labels, confidences = _stabilize_labels(labels, confidences)
    events = _merge_events(labels, confidences, sr, HOP_LENGTH, min_duration)
    return events
