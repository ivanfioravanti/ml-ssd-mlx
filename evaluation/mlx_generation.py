#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Small MLX-LM adapter used by data generation and evaluation."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import mlx.core as mx
from mlx_lm import batch_generate, load
from mlx_lm.sample_utils import make_logits_processors, make_sampler

Prompt = Union[str, Sequence[int]]


@dataclass
class MLXGenerationConfig:
    """Generation settings translated from the repo's vLLM-era config."""

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    repetition_context_size: int = 20
    max_tokens: int = 32768
    seed: Optional[int] = None
    stop: Sequence[str] = ()
    completion_batch_size: int = 32
    prefill_batch_size: int = 8
    prefill_step_size: int = 2048
    max_kv_size: Optional[int] = None

    @classmethod
    def from_sampling_params(
        cls,
        sampling_params: Optional[Dict[str, Any]],
        *,
        max_tokens: int,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> "MLXGenerationConfig":
        params = dict(sampling_params or {})
        params.update(kwargs)
        return cls(max_tokens=max_tokens, seed=seed, **params)


class MLXTextGenerator:
    """Loads an MLX-LM model and generates text from string or token prompts."""

    def __init__(self, model_name: str, *, trust_remote_code: bool = True):
        self.model_name = model_name
        self.model, self.tokenizer = load(
            path_or_hf_repo=model_name,
            tokenizer_config={"trust_remote_code": trust_remote_code},
        )

    def generate(
        self,
        prompts: Sequence[Prompt],
        config: MLXGenerationConfig,
        *,
        verbose: bool = False,
    ) -> List[str]:
        if not prompts:
            return []

        if config.seed is not None:
            mx.random.seed(config.seed)

        self._add_stop_tokens(config.stop)
        token_prompts = [self._encode_prompt(prompt) for prompt in prompts]

        sampler = make_sampler(
            temp=config.temperature,
            top_p=config.top_p,
            min_p=config.min_p,
            top_k=config.top_k,
        )
        logits_processors = make_logits_processors(
            repetition_penalty=config.repetition_penalty,
            repetition_context_size=config.repetition_context_size,
        )

        response = batch_generate(
            self.model,
            self.tokenizer,
            token_prompts,
            max_tokens=config.max_tokens,
            verbose=verbose,
            sampler=sampler,
            logits_processors=logits_processors,
            completion_batch_size=config.completion_batch_size,
            prefill_batch_size=config.prefill_batch_size,
            prefill_step_size=config.prefill_step_size,
            max_kv_size=config.max_kv_size,
        )
        return [self._strip_stop_text(text, config.stop).strip() for text in response.texts]

    def _encode_prompt(self, prompt: Prompt) -> List[int]:
        if isinstance(prompt, str):
            return list(self.tokenizer.encode(prompt))
        return list(prompt)

    def _add_stop_tokens(self, stop: Sequence[str]) -> None:
        add_eos_token = getattr(self.tokenizer, "add_eos_token", None)
        if add_eos_token is None:
            return
        for token in stop:
            try:
                add_eos_token(token)
            except ValueError:
                # Some stop strings are multi-token markers. They are still
                # stripped from decoded text below if they appear.
                continue

    @staticmethod
    def _strip_stop_text(text: str, stop: Sequence[str]) -> str:
        stop_positions = [text.find(stop_text) for stop_text in stop if stop_text]
        stop_positions = [position for position in stop_positions if position >= 0]
        if not stop_positions:
            return text
        return text[: min(stop_positions)]
