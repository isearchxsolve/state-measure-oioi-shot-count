"""Operator CLI.  Mock is the default; live video has two explicit safety gates."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import httpx

from .assembly import HttpClipMaterializer, KenBurnsFallbackMaterializer, SyntheticClipMaterializer, SyntheticFallbackClipMaterializer
from .clients import ChatPoolClient, VideoRouterClient
from .mock import DeterministicMockChatClient, MockSeat, MockVideoBackend, MockVideoConfig
from .orchestrator import FilmOrchestrator, FilmRequest
from .store import FilmJobStore


def _arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Durable long-form film consumer")
    parser.add_argument("--script", required=True, help="Screenplay or generation brief")
    parser.add_argument("--target-seconds", type=int, default=15, help="1800 for a 30-minute film")
    parser.add_argument("--aspect-ratio", default="16:9")
    parser.add_argument("--admission-mode", choices=("refuse", "waves"), default="refuse")
    parser.add_argument("--backend", choices=("mock", "fallback", "live"), default="mock")
    parser.add_argument(
        "--planner", choices=("mock", "pool"), default="mock",
        help="Use deterministic zero-network planning, or the local :20130 crew chat pool.",
    )
    parser.add_argument("--live", action="store_true", help="Required with --backend live")
    parser.add_argument("--max-clips", type=int, default=0, help="Positive hard cap required in live mode")
    parser.add_argument("--ticks", type=int, default=12, help="Bounded scheduler ticks to run after admission")
    parser.add_argument("--tick-sleep", type=float, default=0.0)
    parser.add_argument("--storage", default="./storage/film_pipeline")
    parser.add_argument("--mock-capacity", type=int, default=4)
    return parser


async def _run(args: argparse.Namespace) -> int:
    if args.backend == "live" and (not args.live or args.max_clips <= 0):
        print("REFUSED: live generation requires both --live and --max-clips N (N > 0).", file=sys.stderr)
        return 2
    root = Path(args.storage).resolve()
    store = FilmJobStore(root / "film_jobs.sqlite3")
    owned_mock_http = None
    if args.backend in {"mock", "fallback"}:
        # ``fallback`` deliberately exposes no router seats.  It is the fully local,
        # zero-credit delivery mode: deterministic planning + Pollinations stills +
        # FFmpeg Ken Burns + an AAC Sound Director score.
        fallback_only = args.backend == "fallback"
        backend = MockVideoBackend(MockVideoConfig(seats=[MockSeat("mock-seat-1", clips_remaining=args.mock_capacity)]))
        if fallback_only:
            backend = MockVideoBackend(MockVideoConfig(seats=[]))
        owned_mock_http = httpx.AsyncClient(transport=httpx.ASGITransport(app=backend.app), base_url="http://mock-video")
        chat = (
            ChatPoolClient(os.getenv("FILM_PIPELINE_CHAT_BASE_URL", "http://127.0.0.1:20130"))
            if args.planner == "pool" else DeterministicMockChatClient()
        )
        video = VideoRouterClient(client=owned_mock_http)
        materializer = SyntheticClipMaterializer()
        fallback_materializer = KenBurnsFallbackMaterializer() if fallback_only else SyntheticFallbackClipMaterializer()
    else:
        chat = ChatPoolClient(os.getenv("FILM_PIPELINE_CHAT_BASE_URL", "http://127.0.0.1:20130"))
        video = VideoRouterClient(
            os.getenv("FILM_PIPELINE_VIDEO_BASE_URL", "http://127.0.0.1:8189"),
            mode="live", live=True, max_clips=args.max_clips,
        )
        materializer = HttpClipMaterializer()
        fallback_materializer = KenBurnsFallbackMaterializer()
    pipeline = FilmOrchestrator(
        store=store, chat=chat, video=video, artifact_root=root / "artifacts", materializer=materializer,
        fallback_materializer=fallback_materializer,
        segment_size=2 if args.backend == "mock" else 60,
        execution_backend="live" if args.backend == "live" else "mock",
        live=args.live, max_clips=args.max_clips if args.backend == "live" else None,
    )
    try:
        result = await pipeline.request_film(FilmRequest(
            script=args.script, target_seconds=args.target_seconds, aspect_ratio=args.aspect_ratio,
            admission_mode=args.admission_mode,
            max_clips=args.max_clips if args.backend == "live" else None,
        ))
        if args.backend == "live":
            capacity = await video.capacity()
            provider = capacity.seats[0].provider if capacity.seats else "no provider available"
            cost = result["film"]["cost"]["projected_total"]
            # Required safety receipt, deliberately printed before the first scheduler tick/submit.
            print(f"LIVE PLAN: provider={provider}; clips={result.get('required_unique_clips', 0)}; projected_credits={cost}")
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        if not result.get("accepted", False):
            return 1
        film_id = result["film"]["id"]
        latest = await pipeline.run_until_idle(film_id, max_ticks=max(1, args.ticks), sleep_seconds=max(0.0, args.tick_sleep))
        print(json.dumps(latest, indent=2, ensure_ascii=False, default=str))
        return 0 if latest["film"]["status"] == "completed" else 1
    finally:
        await pipeline.aclose()


def main() -> int:
    return asyncio.run(_run(_arguments().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
