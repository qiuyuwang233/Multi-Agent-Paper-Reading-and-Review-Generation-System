# -*- coding: utf-8 -*-
"""
LangGraph 全局状态定义。

状态在 Planner / Retriever / Reader / Writer / Critic 节点间传递，
用于保存中间结果、回退计数与最终成果物。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from src.schemas import Chunk, Critique, ExtractedInfo, PaperMeta


class WorkflowState(TypedDict, total=False):
    """工作流共享状态。"""

    # 输入
    request: str
    paper_path: str
    language: str
    paper_meta: PaperMeta

    # 中间数据
    chunks: List[Chunk]
    retrieval_queries: List[str]
    retrieved_chunks: Dict[str, List[Chunk]]
    extracted: ExtractedInfo
    tables: List[str]
    critique: Critique
    critique_passed: bool
    critique_suggestions: List[str]
    retry_count: int

    # 输出
    outputs: Dict[str, str]
    output_files: Dict[str, str]

    # 运行信息
    errors: List[str]
    debug: Dict[str, Any]
    finished: bool
    stop_reason: Optional[str]
