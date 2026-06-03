#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Small MLX-LM adapter used by data generation and evaluation."""

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import mlx.core as mx
from mlx_lm import batch_generate, load
from mlx_lm.generate import BatchGenerator
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.utils import sharded_load

Prompt = Union[str, Sequence[int]]


def init_mlx_distributed_group(backend: str = "jaccl") -> Any:
    """Initialize the MLX distributed group used for model-parallel inference."""
    if backend != "jaccl":
        raise ValueError("Only the jaccl backend is supported for MLX distributed.")
    return mx.distributed.init(strict=True, backend=backend)


def distributed_barrier(group: Optional[Any]) -> None:
    """Synchronize distributed ranks when an MLX group is active."""
    if group is None:
        return
    token = mx.distributed.all_sum(mx.array(1, dtype=mx.int32), group=group)
    mx.eval(token)


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
    loop_ngram_min: int = 0
    loop_ngram_max: int = 0
    loop_repetitions: int = 0
    loop_min_tokens: int = 0
    loop_check_interval: int = 128
    loop_text_window_tokens: int = 2048
    loop_text_ngram_min: int = 5
    loop_text_ngram_max: int = 12
    loop_text_repetitions: int = 20
    loop_max_code_fences: int = 0

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

    def __init__(
        self,
        model_name: str,
        *,
        trust_remote_code: bool = True,
        use_mlx_distributed: bool = False,
        distributed_backend: str = "jaccl",
        distributed_group: Optional[Any] = None,
    ):
        self.model_name = model_name
        self.distributed_group = None
        self.distributed_rank = 0
        self.distributed_world_size = 1

        tokenizer_config = {"trust_remote_code": trust_remote_code}
        if use_mlx_distributed:
            self.distributed_group = distributed_group or init_mlx_distributed_group(
                distributed_backend
            )
            self.distributed_rank = self.distributed_group.rank()
            self.distributed_world_size = self.distributed_group.size()
            self.model, self.tokenizer = sharded_load(
                model_name,
                tensor_group=self.distributed_group,
                tokenizer_config=tokenizer_config,
            )
        else:
            self.model, self.tokenizer = load(
                path_or_hf_repo=model_name,
                tokenizer_config=tokenizer_config,
            )
        self.last_finish_reasons: List[str] = []

    @property
    def is_distributed(self) -> bool:
        return self.distributed_group is not None

    @property
    def is_rank0(self) -> bool:
        return self.distributed_rank == 0

    def barrier(self) -> None:
        distributed_barrier(self.distributed_group)

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

        if self._loop_detection_enabled(config):
            texts = self._generate_with_loop_detection(
                token_prompts,
                config,
                sampler=sampler,
                logits_processors=logits_processors,
                verbose=verbose,
            )
        else:
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
            texts = response.texts
            self.last_finish_reasons = ["unknown"] * len(texts)

        return [self._strip_stop_text(text, config.stop).strip() for text in texts]

    def _generate_with_loop_detection(
        self,
        token_prompts: Sequence[List[int]],
        config: MLXGenerationConfig,
        *,
        sampler: Any,
        logits_processors: Any,
        verbose: bool,
    ) -> List[str]:
        gen = BatchGenerator(
            self.model,
            stop_tokens=[[token] for token in self.tokenizer.eos_token_ids],
            sampler=sampler,
            logits_processors=logits_processors,
            completion_batch_size=config.completion_batch_size,
            prefill_batch_size=config.prefill_batch_size,
            prefill_step_size=config.prefill_step_size,
            max_kv_size=config.max_kv_size,
        )

        num_samples = len(token_prompts)
        if verbose:
            print(f"[batch_generate] Finished processing 0/{num_samples} ...", end="\r")

        uids = gen.insert(
            list(token_prompts),
            [config.max_tokens] * len(token_prompts),
        )
        results = {uid: [] for uid in uids}
        finish_reasons = {uid: "" for uid in uids}
        finished = 0

        with gen.stats() as stats:
            while responses := gen.next_generated():
                loop_uids = []
                for response in responses:
                    if response.finish_reason != "stop":
                        results[response.uid].append(response.token)

                    if response.finish_reason is not None:
                        finish_reasons[response.uid] = response.finish_reason
                        finished += 1
                        if verbose:
                            print(
                                f"[batch_generate] Finished processing {finished}/{num_samples} ...",
                                end="\r",
                            )
                    elif self._has_repeated_tail(results[response.uid], config):
                        finish_reasons[response.uid] = "loop"
                        loop_uids.append(response.uid)

                if loop_uids:
                    gen.remove(loop_uids)
                    for _ in loop_uids:
                        finished += 1
                        if verbose:
                            print(
                                f"[batch_generate] Finished processing {finished}/{num_samples} ...",
                                end="\r",
                            )

        gen.close()

        if verbose:
            print(f"[batch_generate] Finished processing {finished}/{num_samples}")
            loop_count = sum(reason == "loop" for reason in finish_reasons.values())
            if loop_count:
                print(f"[batch_generate] Loop-stopped: {loop_count}/{num_samples}")
            print(
                f"[batch_generate] Prompt: {stats.prompt_tokens} tokens, {stats.prompt_tps:.3f} tokens-per-sec"
            )
            print(
                f"[batch_generate] Generation: {stats.generation_tokens} tokens, "
                f"{stats.generation_tps:.3f} tokens-per-sec"
            )
            print(f"[batch_generate] Peak memory: {stats.peak_memory:.3f} GB")

        self.last_finish_reasons = [finish_reasons[uid] or "unknown" for uid in uids]
        return [self.tokenizer.decode(results[uid]) for uid in uids]

    @staticmethod
    def _loop_detection_enabled(config: MLXGenerationConfig) -> bool:
        return (
            config.loop_ngram_min > 0
            and config.loop_ngram_max >= config.loop_ngram_min
            and config.loop_repetitions > 1
        )

    def _has_repeated_tail(self, tokens: Sequence[int], config: MLXGenerationConfig) -> bool:
        if len(tokens) < max(config.loop_min_tokens, config.loop_ngram_min * config.loop_repetitions):
            return False

        max_ngram = min(config.loop_ngram_max, len(tokens) // config.loop_repetitions)
        for ngram_size in range(config.loop_ngram_min, max_ngram + 1):
            tail = tokens[-ngram_size:]
            repeated = True
            for repetition in range(2, config.loop_repetitions + 1):
                start = -repetition * ngram_size
                end = -(repetition - 1) * ngram_size
                if list(tokens[start:end]) != tail:
                    repeated = False
                    break
            if repeated:
                return True

        if config.loop_check_interval <= 0 or len(tokens) % config.loop_check_interval != 0:
            return False

        return self._has_repeated_text_tail(tokens, config)

    def _has_repeated_text_tail(
        self,
        tokens: Sequence[int],
        config: MLXGenerationConfig,
    ) -> bool:
        window_tokens = tokens[-config.loop_text_window_tokens :]
        if not window_tokens:
            return False

        if config.loop_max_code_fences > 0:
            text = self.tokenizer.decode(tokens)
            if text.count("```") >= config.loop_max_code_fences:
                return True
            text = self.tokenizer.decode(window_tokens)
        else:
            text = self.tokenizer.decode(window_tokens)
        words = re.findall(r"[A-Za-z_][A-Za-z_0-9]*|\S", text.lower())
        min_required = config.loop_text_ngram_max + config.loop_text_repetitions
        if len(words) < min_required:
            return False

        max_ngram = min(config.loop_text_ngram_max, len(words))
        for ngram_size in range(config.loop_text_ngram_min, max_ngram + 1):
            ngrams = (
                tuple(words[i : i + ngram_size])
                for i in range(len(words) - ngram_size + 1)
            )
            if Counter(ngrams).most_common(1)[0][1] >= config.loop_text_repetitions:
                return True

        return False

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
