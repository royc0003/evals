"""Retain auditable metadata from vLLM chat completions."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

from aiohttp import ClientResponseError, ClientSession
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from lm_eval.models.api_models import JsonChatStr
from lm_eval.models.openai_completions import LocalChatCompletion

ATTEMPT_SEED_KEY = "_eval_attempt_seed"


@dataclass(frozen=True)
class Completion:
    """Store one logical completion and its transport metadata."""

    text: str
    finish_reason: str | None
    completion_tokens: int | None
    transport_attempts: int
    error: str | None = None

    def __str__(self) -> str:
        """Return response text for lm-eval sample serialization."""
        return self.text


def parse_completions(
    outputs: dict[str, object] | list[dict[str, object]],
) -> list[Completion]:
    """Return completions with the metadata supplied by vLLM."""
    responses = outputs if isinstance(outputs, list) else [outputs]
    completions: list[Completion] = []

    for response in responses:
        raw_usage = response.get("usage")
        raw_choices = response.get("choices")
        if not isinstance(raw_usage, dict) or not isinstance(
            raw_choices, list
        ):
            raise ValueError("vLLM response must contain usage and choices")

        completion_tokens = raw_usage.get("completion_tokens")
        if not isinstance(completion_tokens, int):
            raise ValueError("vLLM response is missing completion token usage")

        choices: list[tuple[int, Completion]] = []
        for raw_choice in raw_choices:
            if not isinstance(raw_choice, dict):
                raise ValueError("vLLM choice must be a mapping")

            index = raw_choice.get("index")
            message = raw_choice.get("message")
            finish_reason = raw_choice.get("finish_reason")
            if not isinstance(index, int) or not isinstance(message, dict):
                raise ValueError("vLLM choice is missing its index or message")

            content = message.get("content")
            if not isinstance(content, str):
                content = ""
            if finish_reason is not None and not isinstance(
                finish_reason, str
            ):
                raise ValueError("vLLM finish reason must be a string or null")

            choices.append(
                (
                    index,
                    Completion(
                        text=content,
                        finish_reason=finish_reason,
                        completion_tokens=completion_tokens,
                        transport_attempts=1,
                    ),
                )
            )

        completions.extend(item for _, item in sorted(choices))

    return completions


def add_attempt_seeds(
    requests: list[Instance],
    base_seed: int,
) -> list[Instance]:
    """Copy requests and assign one deterministic seed per attempt."""
    occurrences: dict[int, int] = {}
    seeded_requests: list[Instance] = []

    for request in requests:
        request_key = id(request)
        attempt_index = occurrences.get(request_key, 0)
        occurrences[request_key] = attempt_index + 1

        context, raw_generation = request.args
        if not isinstance(raw_generation, dict):
            raise TypeError("generation arguments must be a mapping")

        generation = dict(raw_generation)
        generation[ATTEMPT_SEED_KEY] = base_seed + attempt_index
        seeded_requests.append(
            replace(request, arguments=(context, generation))
        )

    return seeded_requests


@register_model("tracked-local-chat-completions")
class TrackedLocalChatCompletion(LocalChatCompletion):  # type: ignore[misc]
    """Add reproducible attempts and response metadata to lm-eval."""

    def __init__(
        self,
        attempts_path: str,
        **kwargs: object,
    ) -> None:
        """Initialize the parent adapter and raw-attempt destination."""
        self.attempts_path = Path(attempts_path)
        super().__init__(**kwargs)

    def _create_payload(
        self,
        messages: list[dict[str, object]],
        generate: bool = False,
        gen_kwargs: dict[str, object] | None = None,
        seed: int = 1234,
        eos: str | None = None,
        **kwargs: object,
    ) -> dict[str, object]:
        """Move the private attempt seed into the public API payload."""
        generation = dict(gen_kwargs or {})
        attempt_seed = generation.pop(ATTEMPT_SEED_KEY, seed)
        if not isinstance(attempt_seed, int):
            raise TypeError("attempt seed must be an integer")

        payload = super()._create_payload(
            messages,
            generate=generate,
            gen_kwargs=generation,
            seed=attempt_seed,
            eos=eos,
            **kwargs,
        )
        return cast(dict[str, object], payload)

    @staticmethod
    def parse_generations(
        outputs: dict[str, object] | list[dict[str, object]],
        **kwargs: object,
    ) -> list[Completion]:
        """Retain vLLM metadata instead of returning response text alone."""
        del kwargs
        return parse_completions(outputs)

    async def amodel_call(
        self,
        session: ClientSession,
        sem: asyncio.Semaphore,
        messages: list[list[int]] | list[str] | list[JsonChatStr],
        *,
        generate: bool = True,
        cache_keys: list[object] | None = None,
        ctxlens: list[int] | None = None,
        gen_kwargs: dict[str, object] | None = None,
        **kwargs: object,
    ) -> list[Completion]:
        """Return one failed completion after bounded transport attempts."""
        if not generate:
            raise NotImplementedError("chat loglikelihood is not supported")

        error: Exception | None = None
        transport_limit = self.max_retries + 1
        for attempt in range(1, transport_limit + 1):
            try:
                result = await super().amodel_call(
                    session,
                    sem,
                    messages,
                    generate=generate,
                    cache_keys=cache_keys,
                    ctxlens=ctxlens,
                    gen_kwargs=gen_kwargs,
                    **kwargs,
                )
                if result is None:
                    raise RuntimeError("vLLM returned no completion")
                completions = cast(list[Completion], result)
                return [
                    replace(item, transport_attempts=attempt)
                    for item in completions
                ]
            except Exception as exc:
                error = exc
                if attempt < transport_limit:
                    await asyncio.sleep(0.5 * 2 ** (attempt - 1))

        if isinstance(error, ClientResponseError):
            message = (
                f"HTTP {error.status} after {transport_limit} "
                "transport attempts"
            )
        else:
            message = (
                f"{type(error).__name__} after {transport_limit} "
                "transport attempts"
            )
        return [
            Completion(
                text="",
                finish_reason=None,
                completion_tokens=None,
                transport_attempts=transport_limit,
                error=message,
            )
        ]

    def generate_until(
        self,
        requests: list[Instance],
        disable_tqdm: bool = False,
    ) -> list[Completion]:
        """Generate seeded completions and record every logical attempt."""
        seeded_requests = add_attempt_seeds(requests, self._seed)
        generated = super().generate_until(
            seeded_requests,
            disable_tqdm=disable_tqdm,
        )
        completions = cast(list[Completion], generated)
        self._write_attempts(seeded_requests, completions)
        return completions

    def _write_attempts(
        self,
        requests: list[Instance],
        completions: list[Completion],
    ) -> None:
        """Write ordered request and response metadata as JSON Lines."""
        records: list[dict[str, object]] = []
        for request, completion in zip(
            requests,
            completions,
            strict=True,
        ):
            context, raw_generation = request.args
            if not isinstance(context, JsonChatStr):
                raise TypeError("chat context must be JSON-formatted messages")
            if not isinstance(raw_generation, dict):
                raise TypeError("generation arguments must be a mapping")

            seed = raw_generation.get(ATTEMPT_SEED_KEY)
            if not isinstance(seed, int):
                raise TypeError("seeded request must contain an integer seed")
            messages = json.loads(context.prompt)
            if not isinstance(messages, list):
                raise TypeError("chat context must contain a message list")

            records.append(
                {
                    "doc_id": request.doc_id,
                    "attempt": seed - self._seed + 1,
                    "seed": seed,
                    "messages": messages,
                    "target": str(request.doc["answer"]),
                    "response": completion.text,
                    "finish_reason": completion.finish_reason,
                    "completion_tokens": completion.completion_tokens,
                    "transport_attempts": completion.transport_attempts,
                    "error": completion.error,
                }
            )

        self.attempts_path.parent.mkdir(parents=True, exist_ok=True)
        content = "".join(
            f"{json.dumps(record, ensure_ascii=False)}\n" for record in records
        )
        self.attempts_path.write_text(content)
