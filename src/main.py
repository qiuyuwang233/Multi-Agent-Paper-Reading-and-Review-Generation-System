# -*- coding: utf-8 -*-
"""
CLI 入口。

示例：
    python -m src.main data/papers/example.pdf --request "关注方法和实验"
    python -m src.main data/papers 2401.00001
    python -m src.main --serve
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

from config.settings import get_settings
from src.graph import run_papers
from src.tools import is_arxiv_input
from src.tools.output_writer import write_batch_comparison, write_outputs


def main() -> None:
    args = _parse_args()
    settings = get_settings()
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
    settings.ensure_dirs()

    if args.serve:
        _run_api(settings.api_host, settings.api_port)
        return

    inputs = _expand_inputs(args.inputs)
    if not inputs:
        raise SystemExit("请提供 PDF 文件、PDF 目录或 arXiv 链接/ID。")

    states = run_papers(inputs, request=args.request, language=args.language)
    for state in states:
        files = write_outputs(state)
        title = state.get("paper_meta").title if state.get("paper_meta") else state.get("paper_path")
        print(f"\n处理完成: {title}")
        if state.get("stop_reason"):
            print("提示:", state["stop_reason"])
        if state.get("critique"):
            print("校验:", state["critique"].summary)
        for key, path in files.items():
            print(f"- {key}: {path}")

    if len(states) > 1:
        batch_path = write_batch_comparison(states)
        print(f"\n批量对比表: {batch_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多 Agent 论文阅读与综述生成系统")
    parser.add_argument("inputs", nargs="*", help="PDF 文件、目录、arXiv ID 或 arXiv 链接")
    parser.add_argument("--request", "-r", default="", help="用户关注点或额外分析要求")
    parser.add_argument("--language", "-l", default="zh", help="输出语言，例如 zh 或 en")
    parser.add_argument("--serve", action="store_true", help="启动 FastAPI 服务")
    return parser.parse_args()


def _expand_inputs(values: List[str]) -> List[str]:
    expanded: List[str] = []
    for value in values:
        if is_arxiv_input(value):
            expanded.append(value)
            continue
        path = Path(value)
        if path.is_dir():
            expanded.extend(str(p) for p in sorted(path.glob("*.pdf")))
        else:
            expanded.append(value)
    return expanded


def _run_api(host: str, port: int) -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("缺少 uvicorn，请先安装 requirements.txt。") from exc
    uvicorn.run("src.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
