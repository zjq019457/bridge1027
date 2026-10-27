#!/usr/bin/env python3
"""
Deterministic chat template for LLaMA-3 Instruct style prompting.

This is used ONLY as a fallback when a local snapshot's tokenizer does not ship
`tokenizer.chat_template` (many offline snapshots omit it).
"""

from __future__ import annotations

# Jinja template compatible with `transformers.PreTrainedTokenizerBase.apply_chat_template`.
#
# Tokens are the standard LLaMA-3 special tokens present in the tokenizer's vocab:
# - <|begin_of_text|>
# - <|start_header_id|> / <|end_header_id|>
# - <|eot_id|>
#
# Note: do not change whitespace/newlines unless you intend to change
# tokenization identity for the entire pipeline.
LLAMA3_FALLBACK_CHAT_TEMPLATE_JINJA: str = (
    "<|begin_of_text|>"
    "{% for message in messages %}"
    "<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n"
    "{{ message['content'] }}<|eot_id|>"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
    "{% endif %}"
)

