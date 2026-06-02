# -*- coding: utf-8 -*-
"""大模型适配层包。"""

from .provider import LLMError, LLMProvider, get_provider

__all__ = ["LLMProvider", "LLMError", "get_provider"]
