"""Typed async HTTP clients for the two frozen supply-layer contracts."""
from __future__ import annotations

import asyncio
from typing import Any, Dict, Literal, Optional

import httpx

from .models import (
    CapacitySeat,
    ChatReply,
    VideoAuthError,
    VideoCapacity,
    VideoJob,
    VideoNoCapacity,
    VideoSubmission,
    VideoTransientError,
    VideoWallError,
    classify_video_failure,
)


class ChatPoolClient:
    """Client for port 20130, deliberately using unpinned ``pool/auto`` by default."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:20130",
        *,
        client: Optional[httpx.AsyncClient] = None,
        retries: int = 3,
    ) -> None:
        self._owned = client is None
        self.client = client or httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=45.0)
        self.retries = max(0, retries)

    async def aclose(self) -> None:
        if self._owned:
            await self.client.aclose()

    async def message(self, content: str, *, max_tokens: int = 1024, model: str = "pool/auto") -> ChatReply:
        """Make one Anthropic-shaped request; only 503 is retried per the contract."""
        payload = {
            "model": model or "pool/auto",
            "max_tokens": max(1, int(max_tokens)),
            "messages": [{"role": "user", "content": content}],
        }
        last: Optional[BaseException] = None
        for attempt in range(self.retries + 1):
            try:
                response = await self.client.post("/v1/messages", json=payload)
                if response.status_code == 503:
                    last = VideoTransientError("chat pool is busy (503)")
                    if attempt < self.retries:
                        await asyncio.sleep(min(12.0, 1.5 * (2**attempt)))
                        continue
                    raise last
                response.raise_for_status()
                body = response.json()
                pieces = body.get("content") or []
                text = "".join(str(p.get("text", "")) for p in pieces if p.get("type") == "text")
                if not text:
                    raise VideoTransientError("chat pool returned no text content")
                return ChatReply(text=text, provider=body.get("x_pool_provider"))
            except httpx.HTTPError as exc:
                last = VideoTransientError(f"chat transport: {type(exc).__name__}")
                if attempt < self.retries:
                    await asyncio.sleep(min(12.0, 1.5 * (2**attempt)))
                    continue
                raise last from exc
        raise VideoTransientError("chat pool request exhausted retries") from last

    async def models(self) -> Dict[str, Any]:
        """Return the chat-pool model catalogue from the frozen ``GET /v1/models`` route."""
        response = await self.client.get("/v1/models")
        response.raise_for_status()
        body = response.json()
        return body if isinstance(body, dict) else {"data": body}

    async def health(self) -> Dict[str, Any]:
        """Return the live pool health document without making a generation request."""
        response = await self.client.get("/health")
        response.raise_for_status()
        body = response.json()
        return body if isinstance(body, dict) else {"status": "unknown", "data": body}


class VideoRouterClient:
    """Client for the frozen asynchronous video router contract on port 8189."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8189",
        *,
        client: Optional[httpx.AsyncClient] = None,
        mode: Literal["mock", "live"] = "mock",
        live: bool = False,
        max_clips: int = 0,
    ) -> None:
        self._owned = client is None
        self.client = client or httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=45.0)
        self.mode = mode
        self.max_clips = int(max_clips)
        self._live_submissions = 0
        if mode == "live" and (not live or self.max_clips <= 0):
            raise ValueError("live video client requires explicit live=True and positive max_clips")

    async def aclose(self) -> None:
        if self._owned:
            await self.client.aclose()

    @staticmethod
    def _error(response: httpx.Response) -> VideoTransientError:
        try:
            message = str((response.json() or {}).get("error") or response.text)
        except ValueError:
            message = response.text
        return VideoTransientError(f"video HTTP {response.status_code}: {message[:240]}")

    async def submit(
        self,
        *,
        job_key: str,
        prompt: str,
        duration_seconds: int,
        image_refs: list[str],
        aspect_ratio: str,
        seed: Optional[int],
    ) -> VideoSubmission:
        if self.mode == "live" and self._live_submissions >= self.max_clips:
            raise VideoTransientError("live video safety cap reached; no further submit is permitted")
        response = await self.client.post(
            "/v1/video/jobs",
            json={
                "jobKey": job_key,
                "prompt": prompt,
                "durationSeconds": int(duration_seconds),
                "imageRefs": image_refs,
                "aspectRatio": aspect_ratio,
                "seed": seed,
            },
        )
        if response.status_code == 503:
            try:
                body = response.json()
            except ValueError:
                body = {}
            if body.get("error") == "no_capacity":
                raise VideoNoCapacity(body.get("retryAfterSeconds", 120))
        if response.status_code not in (200, 202):
            raise self._error(response)
        body = response.json()
        if self.mode == "live" and response.status_code == 202:
            self._live_submissions += 1
        return VideoSubmission(
            job_id=str(body["jobId"]),
            provider=str(body["provider"]),
            estimated_seconds=int(body.get("estimatedSeconds") or 0),
            credits_reserved=float(body.get("creditsReserved") or 0),
            idempotent_replay=response.status_code == 200,
        )

    async def get_job(self, job_id: str) -> VideoJob:
        response = await self.client.get(f"/v1/video/jobs/{job_id}")
        # A router 4xx is a malformed/expired individual poll, not an exception that may
        # escape the scheduler and abort every other film job in this tick.
        if response.status_code < 200 or response.status_code >= 300:
            raise self._error(response)
        body = response.json()
        return VideoJob(
            job_id=str(body["jobId"]),
            status=body["status"],
            provider=str(body.get("provider") or "unknown"),
            progress=float(body.get("progress") or 0.0),
            url=body.get("url"),
            error=body.get("error"),
            credits_spent=float(body.get("creditsSpent") or 0),
        )

    async def capacity(self) -> VideoCapacity:
        response = await self.client.get("/v1/video/capacity")
        # Capacity is an advisory scheduling read.  Treat every non-success response as
        # a retryable router condition so completed clips can still materialise/assemble.
        if response.status_code < 200 or response.status_code >= 300:
            raise self._error(response)
        body = response.json()
        seats = [
            CapacitySeat(
                seat=str(row["seat"]),
                provider=str(row["provider"]),
                credits_remaining=float(row.get("creditsRemaining") or 0),
                clips_remaining=int(row.get("clipsRemaining") or 0),
                cooldown_until=row.get("cooldownUntil"),
            )
            for row in body.get("seats") or []
        ]
        return VideoCapacity(seats=seats, total_clips_available=int(body.get("totalClipsAvailable") or 0))

    async def cancel(self, job_id: str) -> None:
        response = await self.client.delete(f"/v1/video/jobs/{job_id}")
        response.raise_for_status()

    @staticmethod
    def terminal_exception(error: Optional[str]) -> VideoTransientError | VideoWallError | VideoAuthError:
        return classify_video_failure(error)((error or "video job failed").strip())
