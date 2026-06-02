# -*- coding: utf-8 -*-
"""
arXiv 论文拉取工具。

支持输入 arXiv ID、abs 链接或 pdf 链接，将论文 PDF 下载到 data/papers。
"""

from __future__ import annotations

import re
from pathlib import Path

import requests

from config.settings import get_settings

_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(?:v\d+)?")


def is_arxiv_input(value: str) -> bool:
    """判断输入是否像 arXiv ID 或 arXiv 链接。"""
    text = (value or "").strip()
    return "arxiv.org" in text.lower() or bool(_ARXIV_ID_RE.search(text))


def normalize_arxiv_id(value: str) -> str:
    """从 arXiv ID / URL 中提取标准 ID。"""
    text = (value or "").strip()
    match = _ARXIV_ID_RE.search(text)
    if not match:
        raise ValueError(f"无法识别 arXiv ID: {value}")
    return match.group(1)


def download_arxiv_pdf(value: str, output_dir: Path | None = None) -> Path:
    """下载 arXiv PDF，若本地已存在则直接返回。"""
    settings = get_settings()
    arxiv_id = normalize_arxiv_id(value)
    target_dir = output_dir or settings.papers_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"arxiv_{arxiv_id.replace('.', '_')}.pdf"
    if target.exists() and target.stat().st_size > 0:
        return target

    url = f"{settings.arxiv_pdf_base_url.rstrip('/')}/{arxiv_id}.pdf"
    response = requests.get(url, timeout=settings.download_timeout)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if "pdf" not in content_type.lower() and not response.content.startswith(b"%PDF"):
        raise RuntimeError(f"arXiv 返回内容不是 PDF: {url}")

    target.write_bytes(response.content)
    return target
