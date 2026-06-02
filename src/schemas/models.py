# -*- coding: utf-8 -*-
"""
结构化数据模型（基于 pydantic）。

设计要点（防幻觉核心）:
- 每一条被抽取的结论(EvidenceItem)都必须携带 source(SourceRef)，
  即 chunk_id + 页码 + 原文片段，便于 Critic 回溯校验。
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class SourceRef(BaseModel):
    """单条证据来源：将结论映射回原文位置。"""

    chunk_id: str = Field(default="", description="文本块 ID，对应向量库/引用索引")
    page: int = Field(default=0, description="原文页码（从 1 开始）")
    snippet: str = Field(default="", description="支撑该结论的原文片段（用于回溯核对）")


class EvidenceItem(BaseModel):
    """带证据的结构化结论：内容 + 来源列表。"""

    content: str = Field(description="抽取出的结论/要点（基于原文，不得臆造）")
    sources: List[SourceRef] = Field(default_factory=list, description="支撑该结论的证据来源")

    @property
    def verified(self) -> bool:
        """是否至少有一条来源（初步判断，真正核对由 Critic 完成）。"""
        return len(self.sources) > 0


class Chunk(BaseModel):
    """分块后的文本及其定位信息（chunk_id → page/span）。"""

    chunk_id: str
    page: int = 1
    start: int = 0  # 在该页文本中的起始字符偏移
    end: int = 0    # 结束字符偏移
    text: str = ""


class PaperMeta(BaseModel):
    """论文元信息。"""

    paper_id: str = Field(default="", description="论文唯一标识（通常取文件名）")
    title: str = Field(default="", description="论文标题")
    authors: List[str] = Field(default_factory=list, description="作者列表")
    source: str = Field(default="", description="来源（文件路径 / arXiv 链接）")
    n_pages: int = Field(default=0, description="页数")


class ExtractedInfo(BaseModel):
    """Reader 抽取的结构化信息（每项均带证据来源）。"""

    problem: List[EvidenceItem] = Field(default_factory=list, description="问题定义/研究背景")
    method: List[EvidenceItem] = Field(default_factory=list, description="核心方法/框架/关键模块")
    experiments: List[EvidenceItem] = Field(default_factory=list, description="实验设置/数据集/指标/对比结果")
    innovations: List[EvidenceItem] = Field(default_factory=list, description="创新点/贡献")

    def all_items(self) -> List[EvidenceItem]:
        """返回所有字段下的证据条目，便于统一校验。"""
        return [*self.problem, *self.method, *self.experiments, *self.innovations]


class CritiqueIssue(BaseModel):
    """Critic 发现的单条问题。"""

    field: str = Field(description="所属字段：problem/method/experiments/innovations")
    severity: str = Field(default="medium", description="严重级别：high/medium/low")
    description: str = Field(description="问题描述（如：引用不存在、与原文不符）")
    suggestion: str = Field(default="", description="修订建议")


class Critique(BaseModel):
    """Critic 的校验报告。"""

    passed: bool = Field(default=False, description="是否通过校验")
    score: float = Field(default=0.0, description="证据可回溯比例(0~1)")
    issues: List[CritiqueIssue] = Field(default_factory=list, description="发现的问题列表")
    summary: str = Field(default="", description="总体校验结论")


class PaperResult(BaseModel):
    """单篇论文的完整处理结果（贯穿各 Agent 的产物聚合）。"""

    meta: PaperMeta = Field(default_factory=PaperMeta)
    extracted: ExtractedInfo = Field(default_factory=ExtractedInfo)
    critique: Optional[Critique] = None
