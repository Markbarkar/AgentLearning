"""提示词模块"""
from .agent_prompts import get_main_prompt, MAIN_PROMPT_TEMPLATE
from .final_prompts import get_final_prompt, FINAL_PROMPT_TEMPLATE

__all__ = [
    "get_main_prompt",
    "get_final_prompt",
    "MAIN_PROMPT_TEMPLATE",
    "FINAL_PROMPT_TEMPLATE"
]


