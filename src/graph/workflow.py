# -*- coding: utf-8 -*-
"""
LangGraph 工作流：Planner → Retriever → Reader → Writer → Critic。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from config.settings import get_settings
from src.agents import CriticAgent, PlannerAgent, ReaderAgent, RetrieverAgent, WriterAgent
from src.graph.state import WorkflowState
from src.tools import (
    LocalVectorStore,
    download_arxiv_pdf,
    extract_tables_as_markdown,
    is_arxiv_input,
    parse_pdf_to_chunks,
)


class PaperWorkflow:
    """论文处理工作流入口。"""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.planner = PlannerAgent()
        self.retriever = RetrieverAgent()
        self.reader = ReaderAgent()
        self.writer = WriterAgent()
        self.critic = CriticAgent()
        self._compiled = self._compile_graph()

    def run(self, paper_input: str, request: str = "", language: str = "zh") -> WorkflowState:
        """处理单篇论文，返回完整状态。"""
        self.settings.ensure_dirs()
        initial: WorkflowState = {
            "request": request,
            "paper_path": paper_input,
            "language": language,
            "retry_count": 0,
            "errors": [],
            "debug": {},
            "finished": False,
        }
        return self._compiled.invoke(initial)

    def _compile_graph(self):
        try:
            from langgraph.graph import END, StateGraph
        except Exception:
            return _SequentialWorkflow(self)

        graph = StateGraph(WorkflowState)
        graph.add_node("parse", self._parse_node)
        graph.add_node("plan", self._plan_node)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("read", self._read_node)
        graph.add_node("write", self._write_node)
        graph.add_node("critic", self._critic_node)

        graph.set_entry_point("parse")
        graph.add_edge("parse", "plan")
        graph.add_edge("plan", "retrieve")
        graph.add_edge("retrieve", "read")
        graph.add_edge("read", "write")
        graph.add_edge("write", "critic")
        graph.add_conditional_edges(
            "critic",
            self._route_after_critic,
            {"retry": "plan", "done": END},
        )
        return graph.compile()

    def _parse_node(self, state: WorkflowState) -> WorkflowState:
        paper_path = self._resolve_input(state["paper_path"])
        meta, chunks = parse_pdf_to_chunks(str(paper_path))
        tables = extract_tables_as_markdown(str(paper_path))
        return {
            **state,
            "paper_path": str(paper_path),
            "paper_meta": meta,
            "chunks": chunks,
            "tables": tables,
        }

    def _plan_node(self, state: WorkflowState) -> WorkflowState:
        queries = self.planner.plan(state.get("request", ""))
        suggestions = state.get("critique_suggestions") or []
        if suggestions:
            queries.extend([f"根据校验建议补充检索：{s}" for s in suggestions])
        return {**state, "retrieval_queries": queries}

    def _retrieve_node(self, state: WorkflowState) -> WorkflowState:
        meta = state.get("paper_meta")
        collection_name = (
            f"{self.settings.chroma_collection_prefix}_{meta.paper_id}"
            if meta
            else self.settings.chroma_collection_prefix
        )
        vector_store = LocalVectorStore(collection_name=collection_name)
        vector_store.build(state.get("chunks", []))
        retrieved = self.retriever.retrieve(state.get("retrieval_queries", []), vector_store)
        return {**state, "retrieved_chunks": retrieved}

    def _read_node(self, state: WorkflowState) -> WorkflowState:
        extracted = self.reader.extract(
            state.get("retrieved_chunks", {}),
            language=state.get("language", "zh"),
        )
        return {**state, "extracted": extracted}

    def _write_node(self, state: WorkflowState) -> WorkflowState:
        outputs = self.writer.write(
            meta=state["paper_meta"],
            extracted=state["extracted"],
            tables=state.get("tables", []),
            language=state.get("language", "zh"),
        )
        return {**state, "outputs": outputs}

    def _critic_node(self, state: WorkflowState) -> WorkflowState:
        critique = self.critic.review(
            extracted=state["extracted"],
            chunks=state.get("chunks", []),
            outputs=state.get("outputs", {}),
        )
        suggestions = [issue.suggestion for issue in critique.issues if issue.suggestion]
        retry_count = int(state.get("retry_count", 0))
        next_state: WorkflowState = {
            **state,
            "critique": critique,
            "critique_passed": critique.passed,
            "critique_suggestions": suggestions,
        }
        if not critique.passed and retry_count < self.settings.max_retries:
            next_state["retry_count"] = retry_count + 1
            next_state["finished"] = False
            return next_state

        next_state["finished"] = True
        if not critique.passed:
            next_state["stop_reason"] = "达到最大重试次数，已输出当前最佳结果。"
        return next_state

    def _route_after_critic(self, state: WorkflowState) -> str:
        if state.get("critique_passed"):
            return "done"
        if int(state.get("retry_count", 0)) <= self.settings.max_retries and not state.get("finished"):
            return "retry"
        return "done"

    def _resolve_input(self, paper_input: str) -> Path:
        value = (paper_input or "").strip()
        if not value:
            raise ValueError("请输入 PDF 路径或 arXiv 链接/ID。")
        if is_arxiv_input(value):
            return download_arxiv_pdf(value)
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(f"未找到论文文件: {paper_input}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"当前仅支持 PDF 文件: {paper_input}")
        return path


class _SequentialWorkflow:
    """LangGraph 不可用时的顺序执行器，节点逻辑保持一致。"""

    def __init__(self, workflow: PaperWorkflow) -> None:
        self.workflow = workflow

    def invoke(self, state: WorkflowState) -> WorkflowState:
        current = self.workflow._parse_node(state)
        while True:
            current = self.workflow._plan_node(current)
            current = self.workflow._retrieve_node(current)
            current = self.workflow._read_node(current)
            current = self.workflow._write_node(current)
            current = self.workflow._critic_node(current)
            if self.workflow._route_after_critic(current) == "done":
                return current


def run_paper(paper_input: str, request: str = "", language: str = "zh") -> WorkflowState:
    """处理单篇论文的便捷函数。"""
    return PaperWorkflow().run(paper_input=paper_input, request=request, language=language)


def run_papers(
    paper_inputs: Iterable[str],
    request: str = "",
    language: str = "zh",
) -> List[WorkflowState]:
    """批量处理多篇论文。"""
    workflow = PaperWorkflow()
    return [
        workflow.run(paper_input=value, request=request, language=language)
        for value in paper_inputs
    ]
