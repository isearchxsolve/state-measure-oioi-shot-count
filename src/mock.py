"""Configurable zero-credit FastAPI implementation of the frozen video contract."""
from __future__ import annotations

import itertools
import time
import uuid
from dataclasses import dataclass, field
import json
import re
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from .models import ChatReply


@dataclass
class MockSeat:
    seat: str
    provider: str = "mock-video"
    credits_remaining: int = 120
    clips_remaining: int = 4
    cooldown_until: Optional[str] = None


@dataclass
class MockVideoConfig:
    """Injection controls; all values are local-only and never spend credits."""
    latency_seconds: float = 0.0
    seats: List[MockSeat] = field(default_factory=lambda: [MockSeat(seat="mock-seat-1")])
    credits_per_clip: int = 30
    estimated_seconds: int = 1
    #: key -> terminal error.  Prefix ``wall:``/``auth:`` exercises the special paths.
    terminal_failures: Dict[str, str] = field(default_factory=dict)
    #: key -> number of transient HTTP 500 responses before a job is registered.
    submit_transient_failures: Dict[str, int] = field(default_factory=dict)
    #: key -> HTTP status responses injected before a normal status poll.  This
    #: exercises the consumer's non-2xx poll resilience without a real router.
    poll_http_failures: Dict[str, List[int]] = field(default_factory=dict)
    #: One non-2xx capacity response is enough to verify it cannot crash a tick.
    capacity_http_failures: List[int] = field(default_factory=list)


class _CreateJob(BaseModel):
    jobKey: str = Field(min_length=1)
    prompt: str
    durationSeconds: int
    imageRefs: List[str] = Field(default_factory=list)
    aspectRatio: str = "16:9"
    seed: Optional[int] = None


class MockVideoBackend:
    """In-memory mock server exposing the exact frozen routes and response shapes."""

    def __init__(self, config: Optional[MockVideoConfig] = None) -> None:
        self.config = config or MockVideoConfig()
        self.jobs_by_key: Dict[str, dict] = {}
        self.jobs_by_id: Dict[str, dict] = {}
        self._seat_cycle = itertools.cycle(self.config.seats or [MockSeat(seat="mock-seat-1")])
        self.app = FastAPI(title="Mock Frozen Video Router")
        self._mount_routes()

    def set_total_capacity(self, clips: int) -> None:
        """Test control for capacity-wave and exhaustion behavior."""
        remaining = max(0, int(clips))
        for seat in self.config.seats:
            allocation = min(seat.clips_remaining, remaining)
            seat.clips_remaining = allocation
            remaining -= allocation

    def _advance(self, job: dict) -> None:
        if job["status"] not in {"queued", "running"}:
            return
        elapsed = time.monotonic() - job["created_monotonic"]
        latency = max(0.0, self.config.latency_seconds)
        if latency == 0 or elapsed >= latency:
            error = self.config.terminal_failures.get(job["jobKey"])
            job["status"] = "failed" if error else "succeeded"
            job["progress"] = 1.0
            job["error"] = error
            job["url"] = None if error else f"mock://video/{job['jobId']}.mp4"
            job["creditsSpent"] = 0 if error else self.config.credits_per_clip
        else:
            job["status"] = "running"
            job["progress"] = min(0.99, elapsed / latency)

    def _available_seat(self) -> Optional[MockSeat]:
        seats = self.config.seats
        if not seats:
            return None
        for _ in range(len(seats)):
            seat = next(self._seat_cycle)
            if seat.clips_remaining > 0:
                return seat
        return None

    def _mount_routes(self) -> None:
        @self.app.post("/v1/video/jobs")
        async def create_job(payload: _CreateJob, response: Response):
            existing = self.jobs_by_key.get(payload.jobKey)
            if existing is not None:
                response.status_code = 200
                return {
                    "jobId": existing["jobId"], "provider": existing["provider"],
                    "estimatedSeconds": existing["estimatedSeconds"], "creditsReserved": existing["creditsReserved"],
                }
            pending_failures = self.config.submit_transient_failures.get(payload.jobKey, 0)
            if pending_failures > 0:
                self.config.submit_transient_failures[payload.jobKey] = pending_failures - 1
                raise HTTPException(status_code=500, detail="injected transient submit failure")
            seat = self._available_seat()
            if seat is None:
                response.status_code = 503
                return {"error": "no_capacity", "retryAfterSeconds": 1}
            seat.clips_remaining -= 1  # reservation happens on submit, released only by DELETE before start
            job_id = f"mock-{uuid.uuid4().hex[:16]}"
            job = {
                "jobId": job_id, "jobKey": payload.jobKey, "provider": seat.provider, "seat": seat.seat,
                "estimatedSeconds": self.config.estimated_seconds, "creditsReserved": self.config.credits_per_clip,
                "status": "queued", "progress": 0.0, "url": None, "error": None, "creditsSpent": 0,
                "created_monotonic": time.monotonic(), "started": False,
            }
            self.jobs_by_key[payload.jobKey] = job
            self.jobs_by_id[job_id] = job
            response.status_code = 202
            return {
                "jobId": job_id, "provider": job["provider"], "estimatedSeconds": job["estimatedSeconds"],
                "creditsReserved": job["creditsReserved"],
            }

        @self.app.get("/v1/video/jobs/{job_id}")
        async def get_job(job_id: str):
            job = self.jobs_by_id.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="job not found")
            failures = self.config.poll_http_failures.get(job["jobKey"], [])
            if failures:
                raise HTTPException(status_code=failures.pop(0), detail="injected poll transport failure")
            self._advance(job)
            return {
                key: job[key] for key in ("jobId", "status", "provider", "progress", "url", "error", "creditsSpent")
            }

        @self.app.get("/v1/video/capacity")
        async def capacity():
            if self.config.capacity_http_failures:
                raise HTTPException(status_code=self.config.capacity_http_failures.pop(0), detail="injected capacity failure")
            seats = [
                {"seat": s.seat, "provider": s.provider, "creditsRemaining": s.credits_remaining,
                 "clipsRemaining": s.clips_remaining, "cooldownUntil": s.cooldown_until}
                for s in self.config.seats
            ]
            return {"seats": seats, "totalClipsAvailable": sum(s.clips_remaining for s in self.config.seats)}

        @self.app.delete("/v1/video/jobs/{job_id}")
        async def cancel_job(job_id: str):
            job = self.jobs_by_id.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="job not found")
            self._advance(job)
            if job["status"] == "queued":
                for seat in self.config.seats:
                    if seat.seat == job["seat"]:
                        seat.clips_remaining += 1
                        break
                job["status"] = "failed"
                job["error"] = "cancelled"
            return {"jobId": job_id, "status": job["status"]}


def create_mock_video_app(config: Optional[MockVideoConfig] = None) -> FastAPI:
    """Convenience factory for tests or a local ASGI server."""
    return MockVideoBackend(config).app


class DeterministicMockChatClient:
    """Zero-network chat substitute used only for mock pipeline tests/demo runs."""

    async def aclose(self) -> None:
        return None

    async def message(self, content: str, *, max_tokens: int = 1024, model: str = "pool/auto") -> ChatReply:
        if "continuity bible" in content.lower():
            return ChatReply(text=json.dumps({
                "characters": {"lead": "consistent protagonist"}, "wardrobe": "unchanged",
                "locations": ["primary story location"], "lighting": "cinematic continuity",
                "style": "mock cinematic", "imageRefs": [],
            }), provider="mock-chat")
        match = re.search(r"BATCH:\s*shots\s+(\d+)\s+through\s+(\d+)", content, flags=re.IGNORECASE)
        if not match:
            raise ValueError("mock chat could not find requested shot batch")
        first, last = int(match.group(1)), int(match.group(2))
        # Repeat every fourth beat so the mock exercises canonical render reuse deterministically.
        shots = [
            {
                "prompt": f"mock cinematic story beat {(index - 1) % 4 + 1}, stable protagonist and lighting",
                "imageRefs": [],
                "continuity": {"wardrobe": "unchanged", "lighting": "cinematic continuity"},
            }
            for index in range(first, last + 1)
        ]
        return ChatReply(text=json.dumps(shots), provider="mock-chat")
