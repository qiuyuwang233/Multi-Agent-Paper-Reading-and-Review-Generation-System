# -*- coding: utf-8 -*-
"""
全局配置（基于 pydantic-settings）。

所有密钥与模型名集中从项目根目录的 .env 读取，严禁在代码中硬编码。
若 .env 不存在，则回退读取 .env.example，方便快速试跑。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录（config/ 的上一级）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 优先使用 .env，缺失时回退到 .env.example（仅为方便试跑）
_ENV_FILE = PROJECT_ROOT / ".env"
if not _ENV_FILE.exists():
    _ENV_FILE = PROJECT_ROOT / ".env.example"


class Settings(BaseSettings):
    """系统全局配置。"""

    # === DeepSeek 大模型配置 ===
    # 兼容 .env 中的 DEEPSEEK_API_KEY 字段
    deepseek_api_key: str = Field(default="", alias="DEEPSEEK_API_KEY")
    llm_base_url: str = Field(default="https://api.deepseek.com/v1", alias="LLM_BASE_URL")
    llm_model: str = Field(default="deepseek-chat", alias="LLM_MODEL")
    llm_temperature: float = Field(default=0.2, alias="LLM_TEMPERATURE")
    llm_timeout: int = Field(default=120, alias="LLM_TIMEOUT")
    llm_max_retries: int = Field(default=2, alias="LLM_MAX_RETRIES")

    # === Embedding 配置 ===
    embedding_api_key: str = Field(default="", alias="EMBEDDING_API_KEY")
    embedding_base_url: str = Field(default="https://api.deepseek.com/v1", alias="EMBEDDING_BASE_URL")
    embedding_model: str = Field(default="embedding-2", alias="EMBEDDING_MODEL")
    embedding_timeout: int = Field(default=120, alias="EMBEDDING_TIMEOUT")
    embedding_fallback_model: str = Field(
        default="BAAI/bge-small-zh-v1.5",
        alias="EMBEDDING_FALLBACK_MODEL",
    )
    enable_local_embedding_fallback: bool = Field(
        default=True,
        alias="ENABLE_LOCAL_EMBEDDING_FALLBACK",
    )

    # === Chroma 向量库 ===
    chroma_collection_prefix: str = Field(default="paper_chunks", alias="CHROMA_COLLECTION_PREFIX")

    # === 检索 / 分块参数 ===
    chunk_size: int = Field(default=800, alias="CHUNK_SIZE")          # 每个文本块的目标字符数
    chunk_overlap: int = Field(default=120, alias="CHUNK_OVERLAP")    # 相邻块的重叠字符数
    top_k: int = Field(default=5, alias="TOP_K")                      # 每次检索返回的片段数

    # === 校验环路 ===
    max_retries: int = Field(default=2, alias="MAX_RETRIES")          # Critic 触发回退的最大次数

    # === 日志 ===
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # === 路径（相对项目根） ===
    data_dir: str = Field(default="data", alias="DATA_DIR")
    output_dir: str = Field(default="outputs", alias="OUTPUT_DIR")

    # === arXiv 拉取 ===
    arxiv_pdf_base_url: str = Field(default="https://arxiv.org/pdf", alias="ARXIV_PDF_BASE_URL")
    download_timeout: int = Field(default=120, alias="DOWNLOAD_TIMEOUT")

    # === FastAPI ===
    api_host: str = Field(default="127.0.0.1", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # 忽略 .env 中与本配置无关的字段（如 Neo4j / Milvus）
        populate_by_name=True,
    )

    # ---- 派生路径 ----
    @property
    def papers_dir(self) -> Path:
        return PROJECT_ROOT / self.data_dir / "papers"

    @property
    def vector_db_dir(self) -> Path:
        return PROJECT_ROOT / self.data_dir / "vector_db"

    @property
    def outputs_root(self) -> Path:
        return PROJECT_ROOT / self.output_dir

    def ensure_dirs(self) -> None:
        """确保运行所需目录存在。"""
        for sub in ("notes", "related_work", "reviews", "comparison_tables"):
            (self.outputs_root / sub).mkdir(parents=True, exist_ok=True)
        self.papers_dir.mkdir(parents=True, exist_ok=True)
        self.vector_db_dir.mkdir(parents=True, exist_ok=True)

    @property
    def embedding_ready(self) -> bool:
        """是否已配置可用的 Embedding API 密钥。"""
        key = self.embedding_api_key or self.deepseek_api_key
        return bool(key and not key.startswith("请在"))

    @property
    def llm_ready(self) -> bool:
        """是否已配置可用的 LLM 密钥。"""
        return bool(self.deepseek_api_key and not self.deepseek_api_key.startswith("请在"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取全局唯一配置实例（带缓存）。"""
    return Settings()


if __name__ == "__main__":
    # 简单自检：打印关键配置（隐藏密钥）
    s = get_settings()
    masked = (s.deepseek_api_key[:6] + "***") if s.deepseek_api_key else "(未配置)"
    print("配置文件:", _ENV_FILE)
    print("LLM 模型:", s.llm_model, "| 端点:", s.llm_base_url)
    print("Embedding 模型:", s.embedding_model, "| 端点:", s.embedding_base_url)
    print("API Key:", masked, "| 就绪:", s.llm_ready)
    print("项目根:", PROJECT_ROOT)
