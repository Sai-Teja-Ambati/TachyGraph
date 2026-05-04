from __future__ import annotations

import logging
import os
import shutil
import uuid
from pathlib import Path

from tachyrag.config import INGEST_DIR, INGEST_EXTENSIONS
from tachyrag.ingest.ingestor import ingest_document
from tachyrag.ingest.audio_ingestor import AUDIO_EXTENSIONS, ingest_audio

log = logging.getLogger(__name__)


def _extract_pdf_text(filepath: Path) -> str:
    """Extract text from PDF using pymupdf. Falls back gracefully."""
    try:
        import fitz  # pymupdf
    except ImportError:
        raise ValueError("PDF ingestion requires pymupdf: pip install pymupdf")
    doc = fitz.open(str(filepath))
    pages = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            pages.append(text)
    doc.close()
    if not pages:
        raise ValueError(f"No text extracted from {filepath.name} (may be image-only PDF)")
    return "\n\n".join(pages)


def _ensure_dirs() -> tuple[Path, Path]:
    ingest_path = Path(INGEST_DIR)
    done_path = ingest_path / "done"
    failed_path = ingest_path / "failed"
    ingest_path.mkdir(parents=True, exist_ok=True)
    done_path.mkdir(exist_ok=True)
    failed_path.mkdir(exist_ok=True)
    return ingest_path, done_path


def scan_pending() -> list[dict]:
    ingest_path, _ = _ensure_dirs()
    files = []
    for f in sorted(ingest_path.iterdir()):
        if f.is_file() and f.suffix.lower() in INGEST_EXTENSIONS:
            files.append({
                "name": f.name,
                "size_bytes": f.stat().st_size,
                "extension": f.suffix,
            })
    return files


def ingest_file(
    filename: str,
    project_id: uuid.UUID,
    project_name: str = "default",
) -> dict:
    ingest_path, done_path = _ensure_dirs()
    filepath = ingest_path / filename

    if not filepath.exists():
        raise FileNotFoundError(f"{filename} not found in {INGEST_DIR}")
    if not filepath.is_file():
        raise ValueError(f"{filename} is not a file")
    if filepath.suffix.lower() not in INGEST_EXTENSIONS:
        raise ValueError(f"Unsupported extension: {filepath.suffix}")

    if filepath.suffix.lower() == ".pdf":
        text = _extract_pdf_text(filepath)
    elif filepath.suffix.lower() in AUDIO_EXTENSIONS:
        try:
            result = ingest_audio(filepath, project_id, project_name)
            shutil.move(str(filepath), str(done_path / filename))
            result["file"] = filename
            result["status"] = "ingested"
            return result
        except Exception as e:
            failed_path = ingest_path / "failed"
            shutil.move(str(filepath), str(failed_path / filename))
            return {"file": filename, "status": "failed", "error": str(e)}
    else:
        text = filepath.read_text(encoding="utf-8", errors="replace")
    source_url = f"file://{filename}"

    try:
        result = ingest_document(
            text=text,
            source_url=source_url,
            project_id=project_id,
            project_name=project_name,
        )
        # Move to done/ on success
        shutil.move(str(filepath), str(done_path / filename))
        result["file"] = filename
        result["status"] = "ingested"
        log.info("Ingested local file: %s → %d summaries", filename, result["summaries"])
        return result
    except Exception as e:
        # Move to failed/ on error
        failed_path = ingest_path / "failed"
        shutil.move(str(filepath), str(failed_path / filename))
        log.error("Failed to ingest %s: %s", filename, e)
        return {"file": filename, "status": "failed", "error": str(e)}


def ingest_all(
    project_id: uuid.UUID,
    project_name: str = "default",
) -> list[dict]:
    pending = scan_pending()
    results = []
    for f in pending:
        result = ingest_file(f["name"], project_id, project_name)
        results.append(result)
    return results


def ingest_auto() -> list[dict]:
    """Ingest all pending files. Each file gets its own project named after the filename (minus extension)."""
    pending = scan_pending()
    results = []
    for f in pending:
        name = Path(f["name"]).stem.replace("_", " ").replace("-", " ")
        pid = uuid.uuid4()
        result = ingest_file(f["name"], pid, name)
        result["project_id"] = str(pid)
        result["project_name"] = name
        results.append(result)
    return results
