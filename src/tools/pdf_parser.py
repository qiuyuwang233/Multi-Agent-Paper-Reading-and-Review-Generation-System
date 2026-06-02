# -*- coding: utf-8 -*-
"""
PDF 解析工具。

职责：
1. 解析 PDF 文本并按页切分
2. 将文本分块为可检索 Chunk
3. 生成基础论文元信息
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from config.settings import get_settings
from src.schemas import Chunk, PaperMeta


def _safe_import_fitz():
    try:
        import fitz  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("未安装 PyMuPDF，请先执行 `pip install pymupdf`。") from exc
    return fitz


def parse_pdf_pages(pdf_path: str) -> Tuple[PaperMeta, List[Tuple[int, str]]]:
    """读取 PDF，返回论文元信息与逐页文本。"""
    fitz = _safe_import_fitz()
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"未找到论文文件: {pdf_path}")

    doc = fitz.open(str(path))
    pages: List[Tuple[int, str]] = []
    for idx, page in enumerate(doc, start=1):
        text = page.get_text("text") or ""
        pages.append((idx, text.strip()))

    first_page = pages[0][1] if pages else ""
    first_line = next((line.strip() for line in first_page.splitlines() if line.strip()), "")
    title = first_line if first_line else path.stem

    meta = PaperMeta(
        paper_id=path.stem,
        title=title,
        authors=[],
        source=str(path),
        n_pages=len(pages),
    )
    return meta, pages


def chunk_pages(pages: List[Tuple[int, str]], chunk_size: int, chunk_overlap: int) -> List[Chunk]:
    """
    按字符窗口切分文本块，并保留页码与偏移位置。
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0。")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap 不能小于 0。")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size。")

    chunks: List[Chunk] = []
    for page_num, text in pages:
        clean = (text or "").strip()
        if not clean:
            continue

        start = 0
        seq = 0
        n = len(clean)
        while start < n:
            end = min(start + chunk_size, n)
            snippet = clean[start:end]
            chunk_id = f"p{page_num:03d}_c{seq:03d}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    page=page_num,
                    start=start,
                    end=end,
                    text=snippet,
                )
            )
            seq += 1
            if end == n:
                break
            start = max(0, end - chunk_overlap)
    return chunks


def parse_pdf_to_chunks(pdf_path: str) -> Tuple[PaperMeta, List[Chunk]]:
    """一站式接口：PDF -> 元信息 + 分块。"""
    settings = get_settings()
    meta, pages = parse_pdf_pages(pdf_path)
    chunks = chunk_pages(
        pages=pages,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return meta, chunks
