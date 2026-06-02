# 多 Agent 论文阅读与综述生成系统

本项目基于 Planner-Executor-Critic 架构实现论文阅读与综述生成。系统支持 PDF 文件、PDF 目录和 arXiv ID/链接输入，输出四类成果物：

- 论文精读笔记
- Related Work 草稿
- 审稿意见初稿
- 方法对比表

## 环境准备

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

复制 `.env.example` 为 `.env`，填写自己的 API Key：

```powershell
Copy-Item .env.example .env
```

核心配置：

- `DEEPSEEK_API_KEY`：LLM 调用密钥
- `EMBEDDING_API_KEY`：Embedding API 密钥，可与 LLM 密钥相同
- `EMBEDDING_MODEL`：默认 `embedding-2`
- `CHROMA_COLLECTION_PREFIX`：Chroma collection 前缀

## CLI 使用

处理单个 PDF：

```powershell
python -m src.main data\papers\example.pdf --request "重点关注方法和实验"
```

处理目录下所有 PDF：

```powershell
python -m src.main data\papers
```

处理 arXiv 论文：

```powershell
python -m src.main 2401.00001
python -m src.main https://arxiv.org/abs/2401.00001
```

输出会写入：

- `outputs/notes`
- `outputs/related_work`
- `outputs/reviews`
- `outputs/comparison_tables`

## FastAPI 使用

启动服务：

```powershell
python -m src.main --serve
```

接口：

- `GET /health`
- `POST /process`：处理路径或 arXiv 输入
- `POST /upload`：上传 PDF 并处理

`/process` 请求示例：

```json
{
  "inputs": ["2401.00001", "data/papers/example.pdf"],
  "request": "关注方法差异和实验指标"
}
```

## 架构说明

执行链路为：

```text
输入 PDF/arXiv -> PDF 解析 -> Planner -> Retriever -> Reader -> Writer -> Critic -> 输出
```

关键设计：

- Planner 生成检索 query，默认保留确定性任务拆解。
- Retriever 使用 Chroma 持久化向量库和 API Embedding。
- Reader 只基于检索片段抽取结构化字段，每条结论携带 `chunk_id/page/snippet`。
- Writer 仅基于带证据字段生成四类成果物。
- Critic 回查引用，未通过时按 `MAX_RETRIES` 回退重做。

## Windows 注意事项

- 所有源文件按 UTF-8 保存。
- 首次安装 `sentence-transformers` 或本地降级模型时可能需要网络。
- 如果 Chroma 或 API Embedding 不可用，系统会按配置降级到本地模型/哈希向量，保证流程可调试。

## 验证

```powershell
python -m unittest discover -s tests
python -m compileall config src tests
```
# 多 Agent 论文阅读与综述生成系统

基于 Planner-Executor-Critic 架构的论文阅读系统，输入本地 PDF，输出四类成果物：

1. 论文精读笔记（`outputs/notes`）
2. Related Work 草稿（`outputs/related_work`）
3. 审稿意见初稿（`outputs/reviews`）
4. 方法对比表（`outputs/comparison_tables`）

## 技术栈

- Python 3.10+
- LangGraph / LangChain（工作流编排）
- DeepSeek（OpenAI 兼容接口）
- PyMuPDF + pdfplumber（PDF 解析）
- sentence-transformers（向量化，失败时自动降级到哈希向量）

## 快速开始（Windows）

```bash
# 1) 安装依赖
pip install -r requirements.txt

# 2) 配置环境变量
copy .env.example .env
# 然后编辑 .env，填入 DEEPSEEK_API_KEY

# 3) 运行（单篇）
python -m src.main --papers data/papers/your_paper.pdf

# 4) 运行（多篇）
python -m src.main --papers data/papers/a.pdf data/papers/b.pdf --request "关注实验对比和创新点"
```

## 目录说明

```text
MultiAgent/
├── config/
│   └── settings.py
├── src/
│   ├── main.py
│   ├── graph/
│   ├── agents/
│   ├── tools/
│   ├── llm/
│   ├── schemas/
│   └── prompts/
├── data/
│   ├── papers/
│   └── vector_db/
└── outputs/
    ├── notes/
    ├── related_work/
    ├── reviews/
    └── comparison_tables/
```

## 工作流

`Planner -> Retriever -> Reader -> Writer -> Critic`

- Critic 通过：结束并写入成果物
- Critic 不通过：在 `MAX_RETRIES` 限制内回退重试

## 说明

- 所有输出文件均使用 UTF-8 编码。
- Reader 抽取条目默认携带引用信息（`chunk_id/page/snippet`）。
- 若 LLM 不可用，Reader 会降级为规则抽取模式，保证流程可跑通。
