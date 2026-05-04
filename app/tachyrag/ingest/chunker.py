from __future__ import annotations

from dataclasses import dataclass

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from tachyrag.config import CHUNK_OVERLAP, CHUNK_SIZE

_md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ],
    strip_headers=False,
)

_fallback_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


@dataclass
class Chunk:
    content: str
    provenance: dict


def chunk_markdown(text: str, source_url: str) -> list[Chunk]:
    docs = _md_splitter.split_text(text)
    return [
        Chunk(
            content=doc.page_content,
            provenance={
                "source_url": source_url,
                "heading": doc.metadata.get("h1", "")
                + (" > " + doc.metadata["h2"] if "h2" in doc.metadata else "")
                + (" > " + doc.metadata["h3"] if "h3" in doc.metadata else ""),
            },
        )
        for doc in docs
    ]


def chunk_plain(text: str, source_url: str) -> list[Chunk]:
    docs = _fallback_splitter.split_text(text)
    return [
        Chunk(
            content=doc,
            provenance={"source_url": source_url, "heading": ""},
        )
        for doc in docs
    ]


def chunk_document(text: str, source_url: str) -> list[Chunk]:
    if any(text.lstrip().startswith(h) for h in ("#", "##", "###")):
        return chunk_markdown(text, source_url)
    return chunk_plain(text, source_url)
