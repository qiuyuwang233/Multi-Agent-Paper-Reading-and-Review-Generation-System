# -*- coding: utf-8 -*-
"""
Writer Agent：将结构化抽取结果组织成四类成果物。
"""

from __future__ import annotations

from typing import Dict, Iterable, List

from src.schemas import EvidenceItem, ExtractedInfo, PaperMeta


class WriterAgent:
    """基于带证据的字段生成 Markdown 成果物。"""

    def write(
        self,
        meta: PaperMeta,
        extracted: ExtractedInfo,
        tables: List[str] | None = None,
        language: str = "zh",
    ) -> Dict[str, str]:
        """生成 notes / related_work / reviews / comparison_tables 四类内容。"""
        tables = tables or []
        return {
            "notes": self._write_notes(meta, extracted, tables, language),
            "related_work": self._write_related_work(meta, extracted, language),
            "reviews": self._write_review(meta, extracted, language),
            "comparison_tables": self._write_comparison_table(meta, extracted, language),
        }

    def _write_notes(
        self,
        meta: PaperMeta,
        extracted: ExtractedInfo,
        tables: List[str],
        language: str,
    ) -> str:
        text = _texts(language)
        sections = [
            f"# {text['notes_title']}：{meta.title or meta.paper_id}",
            "",
            f"## {text['basic_info']}",
            f"- {text['paper_id']}：{meta.paper_id}",
            f"- {text['source']}：{meta.source}",
            f"- {text['pages']}：{meta.n_pages}",
            "",
            self._section(f"## {text['problem']}", extracted.problem, text),
            self._section(f"## {text['method']}", extracted.method, text),
            self._section(f"## {text['experiments']}", extracted.experiments, text),
            self._section(f"## {text['innovations']}", extracted.innovations, text),
        ]
        if tables:
            sections.extend(["", f"## {text['tables']}", *tables])
        return "\n".join(sections).strip() + "\n"

    def _write_related_work(self, meta: PaperMeta, extracted: ExtractedInfo, language: str) -> str:
        text = _texts(language)
        lines = [
            f"# {text['related_title']}：{meta.title or meta.paper_id}",
            "",
            text["related_intro"],
            "",
            f"## {text['method_context']}",
        ]
        for item in [*extracted.method, *extracted.innovations]:
            lines.append(f"{item.content}{self._cite(item)}")
        if len(lines) <= 5:
            lines.append(text["insufficient_related"])
        return "\n\n".join(lines).strip() + "\n"

    def _write_review(self, meta: PaperMeta, extracted: ExtractedInfo, language: str) -> str:
        text = _texts(language)
        strengths = extracted.innovations or extracted.method
        weaknesses = self._infer_weaknesses(extracted, text)
        lines = [
            f"# {text['review_title']}：{meta.title or meta.paper_id}",
            "",
            f"## {text['overall']}",
            text["overall_text"],
            "",
            self._section(f"## {text['strengths']}", strengths, text),
            f"## {text['weaknesses']}",
            *[f"- {w}" for w in weaknesses],
            "",
            f"## {text['questions']}",
            f"- {text['question_method']}",
            f"- {text['question_experiment']}",
        ]
        return "\n".join(lines).strip() + "\n"

    def _write_comparison_table(self, meta: PaperMeta, extracted: ExtractedInfo, language: str) -> str:
        text = _texts(language)
        method = self._compact_items(extracted.method, text)
        experiments = self._compact_items(extracted.experiments, text)
        innovations = self._compact_items(extracted.innovations, text)
        lines = [
            f"# {text['comparison_title']}：{meta.title or meta.paper_id}",
            "",
            f"| {text['paper']} | {text['method_framework']} | {text['data_metrics']} | {text['innovation']} | {text['evidence']} |",
            "| --- | --- | --- | --- | --- |",
            (
                f"| {meta.title or meta.paper_id} | {method} | {experiments} | "
                f"{innovations} | {self._all_citations(extracted)} |"
            ),
        ]
        return "\n".join(lines).strip() + "\n"

    @staticmethod
    def _section(title: str, items: Iterable[EvidenceItem], text: Dict[str, str]) -> str:
        lines = [title]
        count = 0
        for item in items:
            count += 1
            lines.append(f"- {item.content}{WriterAgent._cite(item)}")
        if count == 0:
            lines.append(f"- {text['empty_item']}")
        return "\n".join(lines)

    @staticmethod
    def _cite(item: EvidenceItem) -> str:
        cites = [
            f"[{source.chunk_id}, p.{source.page}]"
            for source in item.sources
            if source.chunk_id
        ]
        return " " + " ".join(cites) if cites else " [缺少引用]"

    @staticmethod
    def _compact_items(items: Iterable[EvidenceItem], text: Dict[str, str], limit: int = 2) -> str:
        values = [item.content.replace("|", "，").replace("\n", " ") for item in items]
        return "<br>".join(values[:limit]) if values else text["no_evidence"]

    @staticmethod
    def _all_citations(extracted: ExtractedInfo) -> str:
        seen = []
        for item in extracted.all_items():
            for source in item.sources:
                mark = f"{source.chunk_id} p.{source.page}"
                if mark not in seen:
                    seen.append(mark)
        return "<br>".join(seen[:8]) if seen else "缺少引用"

    @staticmethod
    def _infer_weaknesses(extracted: ExtractedInfo, text: Dict[str, str]) -> List[str]:
        weaknesses = []
        if not extracted.experiments:
            weaknesses.append(text["weak_no_experiments"])
        if not extracted.innovations:
            weaknesses.append(text["weak_no_innovations"])
        if not weaknesses:
            weaknesses.append(text["weak_default"])
        return weaknesses


def _texts(language: str) -> Dict[str, str]:
    """返回不同输出语言下的固定文案。"""
    normalized = (language or "zh").lower()
    if normalized in {"en", "english"}:
        return {
            "notes_title": "Reading Notes",
            "basic_info": "Basic Information",
            "paper_id": "Paper ID",
            "source": "Source",
            "pages": "Pages",
            "problem": "Research Problem and Background",
            "method": "Core Method",
            "experiments": "Experimental Results",
            "innovations": "Innovations and Contributions",
            "tables": "Extracted Tables",
            "related_title": "Related Work Draft",
            "related_intro": "This draft is generated only from extracted evidence. Citation markers are kept for manual verification.",
            "method_context": "Method Context",
            "insufficient_related": "Current evidence is insufficient. Add more papers on the same topic for a fuller Related Work section.",
            "review_title": "Review Draft",
            "overall": "Overall Assessment",
            "overall_text": "The following assessment is based on the extracted problem, method, and experimental evidence, and should be manually checked against the paper.",
            "strengths": "Strengths",
            "weaknesses": "Potential Weaknesses",
            "questions": "Questions for the Authors",
            "question_method": "Can the key differences from prior work be further supported by ablation studies?",
            "question_experiment": "Do the datasets, metrics, and baselines cover the main application scenarios?",
            "comparison_title": "Method Comparison Table",
            "paper": "Paper",
            "method_framework": "Method / Framework",
            "data_metrics": "Datasets / Metrics / Results",
            "innovation": "Innovation",
            "evidence": "Evidence",
            "empty_item": "No clear evidence was extracted from the retrieved passages.",
            "no_evidence": "No clear evidence",
            "weak_no_experiments": "The extracted evidence for experimental setup or results is insufficient and should be checked against the original paper.",
            "weak_no_innovations": "The evidence for innovations is insufficient; the differences from prior work need further verification.",
            "weak_default": "The review should still examine experiment coverage, ablation studies, and statistical significance.",
        }
    return {
        "notes_title": "论文精读笔记",
        "basic_info": "基本信息",
        "paper_id": "论文 ID",
        "source": "来源",
        "pages": "页数",
        "problem": "研究问题与背景",
        "method": "核心方法",
        "experiments": "实验结果",
        "innovations": "创新点与贡献",
        "tables": "抽取到的表格",
        "related_title": "Related Work 草稿",
        "related_intro": "以下草稿仅基于当前论文已抽取证据生成，引用标记用于人工回查。",
        "method_context": "方法脉络",
        "insufficient_related": "当前证据不足，建议补充更多同主题论文后生成更完整的 Related Work。",
        "review_title": "审稿意见初稿",
        "overall": "总体评价",
        "overall_text": "该论文的研究目标、方法与实验信息如下，评价需结合原文进一步人工核对。",
        "strengths": "优点",
        "weaknesses": "可能的不足",
        "questions": "给作者的问题",
        "question_method": "方法相对已有工作的关键差异是否可以通过更多消融实验支撑？",
        "question_experiment": "实验设置中的数据集、指标和基线是否覆盖了主要应用场景？",
        "comparison_title": "方法对比表",
        "paper": "论文",
        "method_framework": "方法/框架",
        "data_metrics": "数据集/指标/结果",
        "innovation": "创新点",
        "evidence": "证据",
        "empty_item": "暂未从证据片段中抽取到明确内容。",
        "no_evidence": "暂无明确证据",
        "weak_no_experiments": "当前抽取结果中实验设置或结果证据不足，需要回到原文补充核对。",
        "weak_no_innovations": "创新点证据不足，需进一步确认与已有工作的差异。",
        "weak_default": "审稿时仍需检查实验覆盖范围、消融实验和统计显著性。",
    }
