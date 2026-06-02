# -*- coding: utf-8 -*-
"""
大模型适配层（可切换 Provider）。

默认使用 DeepSeek 的 deepseek-chat（OpenAI 兼容协议），密钥从 .env 注入。
对外暴露统一接口:
    - chat(messages)        : 多轮对话，返回纯文本
    - complete(system,user) : 单轮系统+用户，返回纯文本
    - chat_json(...)        : 要求模型返回 JSON，并稳健解析为 dict

后续如需切换其它厂商，只要其提供 OpenAI 兼容端点，改 .env 即可；
若协议不同，可在此文件新增 Provider 实现而不影响上层 Agent。
"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """LLM 调用相关异常。"""


class LLMProvider:
    """基于 OpenAI 兼容协议的大模型适配器（默认 DeepSeek）。"""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._client = None  # 延迟初始化，避免无密钥时报错

    # ---------------------------------------------------------------
    # 内部：获取 OpenAI 客户端
    # ---------------------------------------------------------------
    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self.settings.llm_ready:
            raise LLMError(
                "未检测到有效的 DEEPSEEK_API_KEY，请在 .env 中正确配置后重试。"
            )
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise LLMError("缺少 openai 依赖，请先 `pip install openai`。") from exc

        self._client = OpenAI(
            api_key=self.settings.deepseek_api_key,
            base_url=self.settings.llm_base_url,
            timeout=self.settings.llm_timeout,
            max_retries=self.settings.llm_max_retries,
        )
        return self._client

    # ---------------------------------------------------------------
    # 基础对话
    # ---------------------------------------------------------------
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """发送多轮消息，返回模型回复的纯文本。"""
        client = self._get_client()
        temp = self.settings.llm_temperature if temperature is None else temperature
        try:
            resp = client.chat.completions.create(
                model=self.settings.llm_model,
                messages=messages,
                temperature=temp,
                max_tokens=max_tokens,
            )
        except Exception as exc:  # 网络/鉴权/限流等
            raise LLMError(f"LLM 调用失败: {exc}") from exc

        content = (resp.choices[0].message.content or "").strip()
        return content

    def complete(self, system: str, user: str, temperature: Optional[float] = None) -> str:
        """单轮系统提示 + 用户输入，返回文本。"""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self.chat(messages, temperature=temperature)

    # ---------------------------------------------------------------
    # JSON 结构化输出
    # ---------------------------------------------------------------
    def chat_json(
        self,
        system: str,
        user: str,
        temperature: Optional[float] = None,
    ) -> Any:
        """
        要求模型输出 JSON，并稳健解析。
        失败时自动追加一次"仅返回合法 JSON"的纠错重试。
        """
        # 在系统提示中强调 JSON 输出
        sys_json = system + "\n\n【输出要求】只输出一个合法的 JSON，不要包含任何解释性文字或 Markdown 代码块标记。"
        raw = self.complete(sys_json, user, temperature=temperature)
        parsed = _try_parse_json(raw)
        if parsed is not None:
            return parsed

        # 纠错重试：把上次输出回灌，要求修正为合法 JSON
        logger.warning("首次 JSON 解析失败，触发纠错重试。")
        fix_user = (
            "你上一次的输出无法被解析为合法 JSON，请仅输出修正后的合法 JSON，"
            "不要添加任何额外文字:\n\n" + raw
        )
        raw2 = self.complete(sys_json, fix_user, temperature=0.0)
        parsed2 = _try_parse_json(raw2)
        if parsed2 is not None:
            return parsed2

        raise LLMError("模型未能返回可解析的 JSON 输出。原始输出片段:\n" + raw[:500])


# -------------------------------------------------------------------
# 工具函数：稳健的 JSON 解析
# -------------------------------------------------------------------
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _try_parse_json(text: str) -> Optional[Any]:
    """从模型输出中尽力提取并解析 JSON 对象/数组。"""
    if not text:
        return None
    candidate = text.strip()

    # 1) 直接解析
    try:
        return json.loads(candidate)
    except Exception:
        pass

    # 2) 去掉 ```json ... ``` 代码块
    m = _FENCE_RE.search(candidate)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            candidate = m.group(1).strip()

    # 3) 截取首个 { 到末个 } （或 [ 到 ]）
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = candidate.find(open_ch)
        end = candidate.rfind(close_ch)
        if 0 <= start < end:
            snippet = candidate[start : end + 1]
            try:
                return json.loads(snippet)
            except Exception:
                continue
    return None


@lru_cache(maxsize=1)
def get_provider() -> LLMProvider:
    """获取全局唯一的 LLM 适配器实例。"""
    return LLMProvider()


if __name__ == "__main__":
    # 简单连通性自检（会真实调用一次 API）
    logging.basicConfig(level=logging.INFO)
    p = get_provider()
    try:
        out = p.complete("你是一个简洁的助手。", "请用一句话介绍论文综述的作用。")
        print("LLM 连通性 OK:\n", out)
    except LLMError as e:
        print("LLM 调用失败:", e)
