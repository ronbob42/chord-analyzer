"""Rich terminal output for chord analysis results."""

import json
import csv
import io
from collections import Counter

from rich.console import Console
from rich.table import Table
from rich.text import Text

from .analyzer import ChordEvent


def _format_time(seconds: float) -> str:
    """Format seconds as M:SS.s"""
    minutes = int(seconds) // 60
    secs = seconds - minutes * 60
    return f"{minutes}:{secs:04.1f}"


def _confidence_bar(confidence: float, width: int = 10) -> Text:
    """Create a colored confidence bar."""
    filled = int(confidence * width)
    empty = width - filled
    pct = int(confidence * 100)

    if confidence >= 0.7:
        color = "green"
    elif confidence >= 0.5:
        color = "yellow"
    else:
        color = "red"

    bar = Text()
    bar.append("█" * filled, style=color)
    bar.append("░" * empty, style="dim")
    bar.append(f" {pct}%", style=color)
    return bar


def _estimate_key(events: list[ChordEvent]) -> str:
    """Heuristic key estimation based on chord root durations."""
    from .templates import NOTE_NAMES

    root_durations: dict[str, float] = {}
    for e in events:
        parts = e.chord.split()
        if len(parts) >= 1:
            root = parts[0]
            root_durations[root] = root_durations.get(root, 0) + e.duration

    if not root_durations:
        return "Unknown"

    # The most prominent root is likely the key
    top_root = max(root_durations, key=root_durations.get)

    # Check if minor chords dominate for that root
    minor_dur = sum(
        e.duration for e in events
        if e.chord.startswith(top_root + " ") and "min" in e.chord
    )
    major_dur = sum(
        e.duration for e in events
        if e.chord.startswith(top_root + " ") and "min" not in e.chord
    )

    quality = "minor" if minor_dur > major_dur else "major"
    return f"{top_root} {quality}"


def display_table(events: list[ChordEvent], metadata: dict, console: Console | None = None):
    """Print a rich table of chord events with summary."""
    if console is None:
        console = Console()

    title = f"{metadata.get('artist', 'Unknown')} - {metadata.get('title', 'Unknown')}"
    table = Table(title=f"Chord Analysis: {title}", show_lines=False)
    table.add_column("#", style="dim", width=4)
    table.add_column("Time", width=10)
    table.add_column("Chord", style="bold cyan", width=10)
    table.add_column("Duration", width=10)
    table.add_column("Confidence", width=18)

    for i, event in enumerate(events, 1):
        time_str = _format_time(event.start_time)
        dur_str = f"{event.duration:.1f}s"
        bar = _confidence_bar(event.confidence)
        table.add_row(str(i), time_str, event.chord, dur_str, bar)

    console.print()
    console.print(table)

    # Summary
    if events:
        chord_counts = Counter(e.chord for e in events)
        total_dur = sum(e.duration for e in events)
        top_chords = chord_counts.most_common(5)
        top_str = ", ".join(
            f"{c} ({d / len(events) * 100:.0f}%)" for c, d in top_chords
        )
        key = _estimate_key(events)

        console.print()
        console.print(f"  [bold]Summary:[/] {len(events)} chord changes | Estimated key: [bold]{key}[/]")
        console.print(f"  [bold]Most common:[/] {top_str}")
        console.print()


def display_json(events: list[ChordEvent], metadata: dict):
    """Print chord events as JSON."""
    data = {
        "metadata": metadata,
        "key_estimate": _estimate_key(events),
        "chords": [
            {
                "chord": e.chord,
                "start": e.start_time,
                "end": e.end_time,
                "duration": round(e.duration, 2),
                "confidence": e.confidence,
            }
            for e in events
        ],
    }
    print(json.dumps(data, indent=2))


def display_csv(events: list[ChordEvent], metadata: dict):
    """Print chord events as CSV."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["start", "end", "duration", "chord", "confidence"])
    for e in events:
        writer.writerow([e.start_time, e.end_time, round(e.duration, 2), e.chord, e.confidence])
    print(output.getvalue(), end="")


def display_simple(events: list[ChordEvent], metadata: dict):
    """Print chord events as plain text."""
    title = f"{metadata.get('artist', 'Unknown')} - {metadata.get('title', 'Unknown')}"
    print(f"\nChord Analysis: {title}\n")
    for i, e in enumerate(events, 1):
        time_str = _format_time(e.start_time)
        print(f"  {i:3d}  {time_str}  {e.chord:<10s}  {e.duration:.1f}s  {int(e.confidence * 100)}%")
    if events:
        key = _estimate_key(events)
        print(f"\n  {len(events)} chord changes | Estimated key: {key}\n")


def display_results(events: list[ChordEvent], metadata: dict, fmt: str = "table", simple: bool = False):
    """Dispatch to the appropriate display function."""
    if simple or fmt == "simple":
        display_simple(events, metadata)
    elif fmt == "json":
        display_json(events, metadata)
    elif fmt == "csv":
        display_csv(events, metadata)
    else:
        display_table(events, metadata)
