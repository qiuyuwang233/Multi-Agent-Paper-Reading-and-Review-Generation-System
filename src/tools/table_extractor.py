# -*- coding: utf-8 -*-
"""
表格抽取工具。

当前实现以 pdfplumber 为主，作为实验结果补充信息来源。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List


def extract_tables_as_markdown(pdf_path: str, max_tables: int = 5) -> List[str]:
    """从 PDF 抽取表格并转成 Markdown。"""
    try:
        import pdfplumber  # type: ignore
    except ImportError:  # pragma: no cover
        return []

    path = Path(pdf_path)
    if not path.exists():
        return []

    tables_md: List[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for pidx, page in enumerate(pdf.pages, start=1):
            extracted = page.extract_tables() or []
            for tidx, table in enumerate(extracted, start=1):
                if not _is_meaningful_table(table):
                    continue
                header = table[0] if table else []
                rows = table[1:] if len(table) > 1 else []
                md = _table_to_md(header, rows, title=f"Page {pidx} Table {tidx}")
                tables_md.append(md)
                if len(tables_md) >= max_tables:
                    return tables_md
    return tables_md


def _table_to_md(header: list, rows: list, title: str = "") -> str:
    safe_header = [_normalize_cell(x) for x in header]
    if not safe_header:
        safe_header = ["col1"]
    lines = []
    if title:
        lines.append(f"### {title}")
    lines.append("| " + " | ".join(safe_header) + " |")
    lines.append("| " + " | ".join(["---"] * len(safe_header)) + " |")
    for row in rows:
        cells = [_normalize_cell(x) for x in row]
        if len(cells) < len(safe_header):
            cells.extend([""] * (len(safe_header) - len(cells)))
        elif len(cells) > len(safe_header):
            cells = cells[: len(safe_header)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _normalize_cell(value: object) -> str:
    """清洗单元格，避免换行和竖线破坏 Markdown 表格。"""
    if value is None:
        return ""
    text = str(value).strip()
    text = " ".join(part.strip() for part in text.splitlines() if part.strip())
    return text.replace("|", "\\|")


def _is_meaningful_table(table: list) -> bool:
    """
    过滤 pdfplumber 误识别出的空表、图例块和过于稀疏的表格。
    """
    if not table or len(table) < 2:
        return False

    normalized_rows = [[_normalize_cell(cell) for cell in row] for row in table if row]
    non_empty_rows = [row for row in normalized_rows if any(cell for cell in row)]
    if len(non_empty_rows) < 2:
        return False

    max_cols = max((len(row) for row in non_empty_rows), default=0)
    if max_cols < 2:
        return False

    non_empty_cells = sum(1 for row in non_empty_rows for cell in row if cell)
    total_cells = len(non_empty_rows) * max_cols
    density = non_empty_cells / total_cells if total_cells else 0
    if non_empty_cells < 4 or density < 0.5:
        return False

    if not _has_text_header(non_empty_rows):
        return False

    # PDF 图例或多列文本被误识别时，常会塞进一个超长单元格。
    if any(len(cell) > 120 for row in non_empty_rows for cell in row):
        return False

    return True


def _has_text_header(rows: list[list[str]]) -> bool:
    """判断表格前两行是否包含可读文本表头。"""
    candidate_rows = rows[:2]
    for row in candidate_rows:
        for cell in row:
            if re.search(r"[A-Za-z\u4e00-\u9fff]", cell):
                return True
    return False
