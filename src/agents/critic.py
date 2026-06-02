# -*- coding: utf-8 -*-
"""
Critic Agent：引用真实性与一致性校验。
"""

from __future__ import annotations

from typing import Dict

from src.schemas import Critique, CritiqueIssue, ExtractedInfo
from src.schemas.models import Chunk
from src.tools.citation import CitationIndex


class CriticAgent:
    """对 Reader/Writer 结果做证据回溯校验。"""

    def review(
        self,
        extracted: ExtractedInfo,
        chunks: list[Chunk],
        outputs: Dict[str, str] | None = None,
    ) -> Critique:
        citation_index = CitationIndex(chunks)
        issues: list[CritiqueIssue] = []
        total_sources = 0
        valid_sources = 0

        for field_name in ("problem", "method", "experiments", "innovations"):
            items = getattr(extracted, field_name)
            if not items:
                issues.append(
                    CritiqueIssue(
                        field=field_name,
                        severity="low",
                        description="该字段未抽取到证据条目。",
                        suggestion="增加相关检索 query 或检查 PDF 解析质量。",
                    )
                )
                continue

            for item in items:
                if not item.sources:
                    issues.append(
                        CritiqueIssue(
                            field=field_name,
                            severity="high",
                            description=f"结论缺少引用来源: {item.content[:80]}",
                            suggestion="为该结论补充 chunk_id、页码和原文片段。",
                        )
                    )
                    continue
                for source in item.sources:
                    total_sources += 1
                    ok, reason = citation_index.verify_source(source)
                    if ok:
                        valid_sources += 1
                    else:
                        issues.append(
                            CritiqueIssue(
                                field=field_name,
                                severity="high",
                                description=reason,
                                suggestion="删除该结论或重新从原文片段抽取。",
                            )
                        )

        if outputs:
            for name, content in outputs.items():
                if content and "[缺少引用]" in content:
                    issues.append(
                        CritiqueIssue(
                            field=name,
                            severity="medium",
                            description="输出中存在缺少引用的内容。",
                            suggestion="仅保留带证据来源的结论。",
                        )
                    )

        score = valid_sources / total_sources if total_sources else 0.0
        has_high = any(issue.severity == "high" for issue in issues)
        passed = score >= 0.8 and not has_high
        summary = (
            f"引用校验通过率 {score:.0%}，共发现 {len(issues)} 个问题。"
            if issues
            else "所有抽取条目均可回溯到原文片段。"
        )
        return Critique(passed=passed, score=score, issues=issues, summary=summary)
