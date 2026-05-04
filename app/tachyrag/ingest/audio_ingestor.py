from __future__ import annotations

import logging
import uuid
from pathlib import Path

log = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm"}


def transcribe_audio(filepath: Path) -> str:
    """Transcribe audio file to text using faster-whisper."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ValueError("Audio ingestion requires faster-whisper: pip install faster-whisper")

    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, info = model.transcribe(str(filepath), beam_size=5)

    text_parts = []
    for segment in segments:
        text_parts.append(segment.text.strip())

    full_text = " ".join(text_parts)
    log.info("Transcribed %s: %d chars, language=%s", filepath.name, len(full_text), info.language)

    if not full_text.strip():
        raise ValueError(f"No speech detected in {filepath.name}")

    return full_text


def ingest_audio(
    filepath: Path,
    project_id: uuid.UUID,
    project_name: str = "default",
) -> dict:
    """Transcribe audio and ingest the text."""
    from tachyrag.ingest.ingestor import ingest_document

    text = transcribe_audio(filepath)
    return ingest_document(
        text=text,
        source_url=f"audio://{filepath.name}",
        project_id=project_id,
        project_name=project_name,
    )
