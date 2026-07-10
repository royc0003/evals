"""Test the narrow lm-eval adapter for the vLLM endpoint."""

from __future__ import annotations

import asyncio
import json
from dataclasses import FrozenInstanceError
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import import_module
from pathlib import Path
from threading import Thread

import pytest
from aiohttp import ClientSession, web
from aiohttp.test_utils import TestServer
from lm_eval.api.instance import Instance
from lm_eval.models.api_models import JsonChatStr


def test_completion_retains_auditable_response_fields() -> None:
    """Retain the response fields required by grading and reporting."""
    adapter = import_module("evals.vllm_adapter")
    completion = adapter.Completion(
        text="Exact Answer: 42",
        finish_reason="stop",
        completion_tokens=123,
        transport_attempts=1,
    )

    assert completion.text == "Exact Answer: 42"
    assert completion.finish_reason == "stop"
    assert completion.completion_tokens == 123
    assert completion.transport_attempts == 1
    assert completion.error is None
    assert str(completion) == "Exact Answer: 42"
    with pytest.raises(FrozenInstanceError):
        completion.text = "changed"


def test_parse_completions_retains_vllm_metadata() -> None:
    """Parse response text, finish reason, and completion token usage."""
    adapter = import_module("evals.vllm_adapter")
    response = {
        "choices": [
            {
                "index": 0,
                "message": {"content": "Exact Answer: 42"},
                "finish_reason": "length",
            }
        ],
        "usage": {"completion_tokens": 321},
    }

    assert adapter.parse_completions(response) == [
        adapter.Completion(
            text="Exact Answer: 42",
            finish_reason="length",
            completion_tokens=321,
            transport_attempts=1,
        )
    ]

    assert adapter.TrackedLocalChatCompletion.parse_generations(response) == [
        adapter.Completion(
            text="Exact Answer: 42",
            finish_reason="length",
            completion_tokens=321,
            transport_attempts=1,
        )
    ]


def test_add_attempt_seeds_assigns_16_seeds_per_question() -> None:
    """Assign the same reproducible 16-seed schedule to each question."""
    adapter = import_module("evals.vllm_adapter")
    first = Instance(
        request_type="generate_until",
        doc={"problem": "first", "answer": "1"},
        arguments=("first prompt", {"temperature": 1.0}),
        idx=0,
        metadata=("aime26", 0, 16),
    )
    second = Instance(
        request_type="generate_until",
        doc={"problem": "second", "answer": "2"},
        arguments=("second prompt", {"temperature": 1.0}),
        idx=0,
        metadata=("aime26", 1, 16),
    )
    cloned = [first] * 16 + [second] * 16

    seeded = adapter.add_attempt_seeds(cloned, base_seed=2026)

    assert [
        item.args[1][adapter.ATTEMPT_SEED_KEY] for item in seeded[:16]
    ] == list(range(2026, 2042))
    assert [
        item.args[1][adapter.ATTEMPT_SEED_KEY] for item in seeded[16:]
    ] == list(range(2026, 2042))
    assert first.args[1] == {"temperature": 1.0}
    assert second.args[1] == {"temperature": 1.0}
    assert all(item is not first and item is not second for item in seeded)


def test_create_payload_sends_only_the_public_attempt_seed(
    tmp_path: Path,
) -> None:
    """Send the attempt seed without leaking the adapter's private key."""
    adapter = import_module("evals.vllm_adapter")
    model = adapter.TrackedLocalChatCompletion(
        model="qwen3.5-9b",
        base_url="http://example.test/v1/chat/completions",
        tokenizer_backend=None,
        tokenized_requests=False,
        attempts_path=str(tmp_path / "attempts.jsonl"),
        max_gen_toks=100,
    )
    generation = {
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 0.95,
        "max_gen_toks": 50,
        adapter.ATTEMPT_SEED_KEY: 2029,
    }

    payload = model._create_payload(
        [{"role": "user", "content": "Solve this."}],
        generate=True,
        gen_kwargs=generation,
    )

    assert payload["seed"] == 2029
    assert payload["max_tokens"] == 50
    assert adapter.ATTEMPT_SEED_KEY not in payload
    assert adapter.ATTEMPT_SEED_KEY in generation


def test_failed_transport_becomes_one_failed_completion(
    tmp_path: Path,
) -> None:
    """Keep one logical attempt after an initial call and three retries."""
    adapter = import_module("evals.vllm_adapter")
    transport_calls = 0

    async def exercise() -> object:
        async def unavailable(_: web.Request) -> web.Response:
            nonlocal transport_calls
            transport_calls += 1
            return web.json_response({"error": "unavailable"}, status=503)

        application = web.Application()
        application.router.add_post("/chat/completions", unavailable)
        server = TestServer(application)
        await server.start_server()
        model = adapter.TrackedLocalChatCompletion(
            model="qwen3.5-9b",
            base_url=str(server.make_url("/chat/completions")),
            tokenizer_backend=None,
            tokenized_requests=False,
            attempts_path=str(tmp_path / "attempts.jsonl"),
            max_retries=3,
        )
        messages = [
            JsonChatStr(
                json.dumps([{"role": "user", "content": "Solve this."}])
            )
        ]
        try:
            async with ClientSession() as session:
                return await model.amodel_call(
                    session,
                    asyncio.Semaphore(1),
                    messages,
                    gen_kwargs={adapter.ATTEMPT_SEED_KEY: 2026},
                )
        finally:
            await server.close()

    result = asyncio.run(exercise())

    assert transport_calls == 4
    assert isinstance(result, list)
    assert result == [
        adapter.Completion(
            text="",
            finish_reason=None,
            completion_tokens=None,
            transport_attempts=4,
            error="HTTP 503 after 4 transport attempts",
        )
    ]


def test_generate_until_seeds_and_records_each_logical_attempt(
    tmp_path: Path,
) -> None:
    """Send seeded requests and write their auditable response metadata."""
    adapter = import_module("evals.vllm_adapter")
    payloads: list[dict[str, object]] = []

    class Handler(BaseHTTPRequestHandler):
        """Return one deterministic completion for each received seed."""

        def do_POST(self) -> None:
            """Capture the payload and return a chat completion."""
            length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(length))
            assert isinstance(payload, dict)
            payloads.append(payload)
            body = json.dumps(
                {
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "content": f"Exact Answer: {payload['seed']}"
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"completion_tokens": 5},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            """Suppress the test server access log."""
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever)
    thread.start()
    messages = [
        {"role": "system", "content": "Use the required fields."},
        {"role": "user", "content": "Problem text"},
    ]
    instance = Instance(
        request_type="generate_until",
        doc={"problem": "Problem text", "answer": "7"},
        arguments=(
            JsonChatStr(json.dumps(messages)),
            {
                "do_sample": True,
                "temperature": 1.0,
                "max_gen_toks": 100,
            },
        ),
        idx=0,
        metadata=("aime26", 4, 2),
    )
    attempts_path = tmp_path / "attempts.jsonl"
    model = adapter.TrackedLocalChatCompletion(
        model="qwen3.5-9b",
        base_url=(f"http://127.0.0.1:{server.server_port}/chat/completions"),
        tokenizer_backend=None,
        tokenized_requests=False,
        attempts_path=str(attempts_path),
        num_concurrent=2,
        seed=2026,
    )

    try:
        completions = model.generate_until([instance, instance])
    finally:
        server.shutdown()
        thread.join()
        server.server_close()

    seeds: list[int] = []
    for payload in payloads:
        seed = payload["seed"]
        assert isinstance(seed, int)
        seeds.append(seed)
    assert sorted(seeds) == [2026, 2027]
    assert [completion.text for completion in completions] == [
        "Exact Answer: 2026",
        "Exact Answer: 2027",
    ]
    records = [
        json.loads(line) for line in attempts_path.read_text().splitlines()
    ]
    assert records == [
        {
            "doc_id": 4,
            "attempt": 1,
            "seed": 2026,
            "messages": messages,
            "target": "7",
            "response": "Exact Answer: 2026",
            "finish_reason": "stop",
            "completion_tokens": 5,
            "transport_attempts": 1,
            "error": None,
        },
        {
            "doc_id": 4,
            "attempt": 2,
            "seed": 2027,
            "messages": messages,
            "target": "7",
            "response": "Exact Answer: 2027",
            "finish_reason": "stop",
            "completion_tokens": 5,
            "transport_attempts": 1,
            "error": None,
        },
    ]
