# -*- coding: utf-8 -*-
"""
FastAPI 接口入口。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from config.settings import get_settings
from config.settings import PROJECT_ROOT
from src.graph import run_paper, run_papers
from src.tools.output_writer import write_batch_comparison, write_outputs

app = FastAPI(title="多 Agent 论文阅读与综述生成系统", version="1.0.0")


class ProcessRequest(BaseModel):
    """路径或 arXiv 输入请求。"""

    inputs: List[str] = Field(description="PDF 路径、arXiv ID 或 arXiv 链接")
    request: str = Field(default="", description="用户关注点")
    language: str = Field(default="zh", description="输出语言，如 zh/en")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """返回前端交互页面。"""
    index_path = PROJECT_ROOT / "src" / "web" / "index.html"
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.post("/process")
def process(req: ProcessRequest) -> Dict[str, object]:
    """处理一个或多个路径/arXiv 输入。"""
    states = run_papers(req.inputs, request=req.request, language=req.language)
    results = [_state_to_response(state) for state in states]
    batch_file = write_batch_comparison(states) if len(states) > 1 else ""
    return {"results": results, "batch_comparison": batch_file}


@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    request: str = Form(default=""),
    language: str = Form(default="zh"),
) -> Dict[str, object]:
    """上传单个 PDF 并处理。"""
    settings = get_settings()
    settings.ensure_dirs()
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return {"error": "请上传 PDF 文件。"}

    target = settings.papers_dir / Path(file.filename).name
    target.write_bytes(await file.read())
    state = run_paper(str(target), request=request, language=language)
    return _state_to_response(state)


def _state_to_response(state) -> Dict[str, object]:
    files = write_outputs(state)
    meta = state.get("paper_meta")
    critique = state.get("critique")
    return {
        "paper_id": meta.paper_id if meta else "",
        "title": meta.title if meta else "",
        "output_files": files,
        "outputs": state.get("outputs", {}),
        "critique": critique.model_dump() if critique else None,
        "stop_reason": state.get("stop_reason", ""),
    }
