# -*- coding: utf-8 -*-
"""
Planner Agent：任务拆解与调度规划。
"""

from __future__ import annotations

from typing import List

from src.llm import LLMError, get_provider


class PlannerAgent:
    """根据用户请求生成子任务检索 query 列表。"""

    DEFAULT_QUERIES = [
        "论文要解决的核心问题、研究背景与动机",
        "方法框架、关键模块、公式或算法流程",
        "实验设置、数据集、评价指标、对比基线与结果",
        "论文创新点、贡献和与已有工作的差异",
    ]

    def plan(self, request: str) -> List[str]:
        """
        优先尝试 LLM 细化任务；失败时保留确定性策略，保证可解释、可复现。
        """
        req = (request or "").strip()
        if req:
            try:
                data = get_provider().chat_json(
                    system="你是论文阅读任务规划器。请将用户目标拆成 4 到 8 条中文检索 query。",
                    user=f'用户目标: {req}\n请返回 JSON: {{"queries":["..."]}}',
                    temperature=0.1,
                )
                queries = data.get("queries", []) if isinstance(data, dict) else []
                clean = [str(q).strip() for q in queries if str(q).strip()]
                if clean:
                    return [*self.DEFAULT_QUERIES, *clean]
            except (LLMError, Exception):
                pass

        if not req:
            return self.DEFAULT_QUERIES.copy()
        # 保留默认任务，并将用户请求拼接为额外聚焦任务。
        return [*self.DEFAULT_QUERIES, f"结合用户关注点补充分析：{req}"]
