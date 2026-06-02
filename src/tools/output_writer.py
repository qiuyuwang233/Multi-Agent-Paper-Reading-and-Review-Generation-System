# -*- coding: utf-8 -*-
"""
成果物写入工具。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable

from config.settings import get_settings
from src.graph.state import WorkflowState

OUTPUT_SUBDIRS = {
    "notes": "notes",
    "related_work": "related_work",
    "reviews": "reviews",
    "comparison_tables": "comparison_tables",
}


def write_outputs(state: WorkflowState) -> Dict[str, str]:
    """将单篇论文四类成果物写入 outputs 目录。"""
    settings = get_settings()
    settings.ensure_dirs()
    meta = state.get("paper_meta")
    paper_id = _safe_name(meta.paper_id if meta else Path(state.get("paper_path", "paper")).stem)
    output_files: Dict[str, str] = {}

    for key, content in (state.get("outputs") or {}).items():
        subdir = OUTPUT_SUBDIRS.get(key)
        if not subdir:
            continue
        suffix = ".csv" if key == "comparison_tables" and content.lstrip().startswith("paper,") else ".md"
        path = settings.outputs_root / subdir / f"{paper_id}_{key}{suffix}"
        path.write_text(content, encoding="utf-8")
        output_files[key] = str(path)

    state["output_files"] = output_files
    return output_files


def write_batch_comparison(states: Iterable[WorkflowState], filename: str = "batch_comparison.md") -> str:
    """汇总多篇论文的对比表，便于横向查看。"""
    settings = get_settings()
    settings.ensure_dirs()
    rows = [
        "# 多论文方法对比表",
        "",
        "| 论文 | 方法/框架 | 数据集/指标/结果 | 创新点 | 证据 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for state in states:
        table = (state.get("outputs") or {}).get("comparison_tables", "")
        for line in table.splitlines():
            if line.startswith("| ") and not line.startswith("| ---") and "论文 | 方法" not in line:
                rows.append(line)

    path = settings.outputs_root / "comparison_tables" / filename
    path.write_text("\n".join(rows).strip() + "\n", encoding="utf-8")
    return str(path)


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff_-]+", "_", value or "paper").strip("_")
    return cleaned[:80] or "paper"
