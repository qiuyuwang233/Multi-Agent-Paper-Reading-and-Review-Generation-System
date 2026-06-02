# -*- coding: utf-8 -*-
"""基础单元测试。"""

from __future__ import annotations

import unittest

from src.agents.critic import CriticAgent
from src.agents.writer import WriterAgent
from src.schemas import Chunk, EvidenceItem, ExtractedInfo, PaperMeta, SourceRef
from src.tools.citation import CitationIndex
from src.tools.pdf_parser import chunk_pages


class BasicWorkflowPartsTest(unittest.TestCase):
    def test_chunk_pages_keeps_source_offsets(self) -> None:
        chunks = chunk_pages([(1, "abcdefg")], chunk_size=4, chunk_overlap=1)
        self.assertEqual([c.text for c in chunks], ["abcd", "defg"])
        self.assertEqual(chunks[0].chunk_id, "p001_c000")
        self.assertEqual(chunks[1].start, 3)

    def test_citation_index_verifies_source(self) -> None:
        chunk = Chunk(chunk_id="p001_c000", page=1, start=0, end=10, text="核心方法来自原文。")
        index = CitationIndex([chunk])
        ok, reason = index.verify_source(
            SourceRef(chunk_id="p001_c000", page=1, snippet="核心方法")
        )
        self.assertTrue(ok, reason)

    def test_writer_and_critic_use_evidence(self) -> None:
        chunk = Chunk(chunk_id="p001_c000", page=1, start=0, end=20, text="本文提出一个多 Agent 方法。")
        source = SourceRef(chunk_id=chunk.chunk_id, page=1, snippet="多 Agent 方法")
        extracted = ExtractedInfo(
            method=[EvidenceItem(content="本文提出一个多 Agent 方法。", sources=[source])],
            experiments=[EvidenceItem(content="实验显示方法有效。", sources=[source])],
            innovations=[EvidenceItem(content="创新点是多角色协作。", sources=[source])],
        )
        meta = PaperMeta(paper_id="demo", title="Demo Paper", source="demo.pdf", n_pages=1)
        outputs = WriterAgent().write(meta, extracted)
        critique = CriticAgent().review(extracted, [chunk], outputs)
        self.assertIn("notes", outputs)
        self.assertGreaterEqual(critique.score, 0.8)


if __name__ == "__main__":
    unittest.main()
