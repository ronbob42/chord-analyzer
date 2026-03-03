#!/usr/bin/env python3
"""Spotify Chord Analyzer — detect chords from any Spotify track."""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from rich.console import Console

from chord_analyzer.downloader import download_track, DownloadError
from chord_analyzer.analyzer import analyze_audio
from chord_analyzer.display import display_results
from chord_analyzer.player import launch_player


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze chords in a Spotify track.",
        epilog="Example: python main.py https://open.spotify.com/track/4sdhaq0Z4KL5oCF5GCgTtt",
    )
    parser.add_argument("spotify_url", help="Spotify track URL")
    parser.add_argument(
        "-f", "--format",
        choices=["table", "json", "csv"],
        default="table",
        help="Output format (default: table)",
    )
    parser.add_argument(
        "-s", "--simple",
        action="store_true",
        help="Plain text output (no rich formatting)",
    )
    parser.add_argument(
        "-k", "--keep-audio",
        action="store_true",
        help="Keep the downloaded audio file after analysis",
    )
    parser.add_argument(
        "-o", "--output-dir",
        help="Directory for downloaded audio (default: temp dir)",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.3,
        help="Minimum chord duration in seconds (default: 0.3)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.4,
        help="Minimum confidence threshold (default: 0.4)",
    )
    parser.add_argument(
        "-p", "--play",
        action="store_true",
        help="Open a web-based chord player in the browser after analysis",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    console = Console()
    temp_dir = None

    try:
        # Determine output directory
        if args.output_dir:
            out_dir = args.output_dir
        else:
            temp_dir = tempfile.mkdtemp(prefix="chord_analyzer_")
            out_dir = temp_dir

        # Download
        with console.status("[bold blue]Downloading audio from Spotify..."):
            audio_path, metadata = download_track(args.spotify_url, out_dir)
        console.print(f"  [green]Downloaded:[/] {metadata['artist']} - {metadata['title']}")

        # Analyze
        with console.status("[bold blue]Analyzing chords..."):
            events = analyze_audio(
                audio_path,
                min_duration=args.min_duration,
                confidence_threshold=args.confidence,
            )

        if not events:
            console.print("[yellow]No chords detected in this track.[/]")
            return

        # Display
        display_results(events, metadata, fmt=args.format, simple=args.simple)

        # Play mode — open web player in browser
        if args.play:
            launch_player(events, metadata, audio_path)

        if args.keep_audio:
            console.print(f"  [dim]Audio saved to: {audio_path}[/]")

    except ValueError as e:
        console.print(f"[red]Error:[/] {e}")
        sys.exit(1)
    except DownloadError as e:
        console.print(f"[red]Download error:[/] {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled.[/]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/] {e}")
        sys.exit(1)
    finally:
        # Cleanup temp files unless user wants to keep them
        if not args.keep_audio and temp_dir and Path(temp_dir).exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
