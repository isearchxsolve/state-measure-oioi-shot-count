"""Admission-controlled, durable asynchronous film scheduler.

The scheduler treats the async video router as a quality layer, never as a condition
for completing an edit. A router capacity wall produces a locally rendered Ken Burns
slot with the same timing and timeline placement as a remote motion clip.
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from studio_pipeline import media

from .assembly import (
    ClipMaterializer,
    FallbackClipMaterializer,
    HttpClipMaterializer,
    IncrementalAssembler,
    KenBurnsFallbackMaterializer,
)
from .clients import ChatPoolClient, VideoRouterClient
from .models import VideoAuthError, VideoNoCapacity, VideoTransientError, VideoWallError
from .shots import ShotListPlanner, job_key_for_shot, request_key_for_film
from .store import FilmJobStore


AdmissionMode = Literal["refuse", "waves"]


@dataclass(frozen=True)
class FilmRequest:
    script: str
    target_seconds: int = 10_800
    aspect_ratio: str = "16:9"
    style: str = "cinematic"
    image_refs: List[str] = field(default_factory=list)
    seed: Optional[int] = 20_260_826
    bible: Optional[Dict[str, Any]] = None
    admission_mode: AdmissionMode = "refuse"
    #: Explicit hard cap for router submissions in live mode, not local fallback shots.
    max_clips: Optional[int] = None
    #: The normal mode: a complete MP4 never depends on provider capacity.
    allow_fallback: bool = True
    #: Sound Director music is muxed into the final MP4.
    with_audio: bool = True


class FilmOrchestrator:
    """The consumption-layer scheduler; safe to reconstruct after a process restart."""

    _TERMINAL_FILM_STATES = {
        "completed", "failed", "assembly_error", "refused_capacity", "refused_safety_cap",
    }

    def __init__(
        self,
        *,
        store: FilmJobStore,
        chat: ChatPoolClient,
        video: VideoRouterClient,
        artifact_root: Path | str,
        materializer: Optional[ClipMaterializer] = None,
        fallback_materializer: Optional[FallbackClipMaterializer] = None,
        per_provider_cap: int = 1,
        per_seat_cap: int = 1,
        max_attempts: int = 3,
        poll_interval_seconds: float = 5.0,
        segment_size: int = 60,
        materialize_concurrency: int = 4,
        fallback_concurrency: int = 4,
        submit_concurrency: int = 16,
        execution_backend: Literal["mock", "live"] = "mock",
        live: bool = False,
        max_clips: Optional[int] = None,
    ) -> None:
        self.store = store
        self.chat = chat
        self.video = video
        self.planner = ShotListPlanner(chat)
        self.materializer = materializer or HttpClipMaterializer()
        self.fallback_materializer = fallback_materializer or KenBurnsFallbackMaterializer()
        self.assembler = IncrementalAssembler(store, artifact_root, segment_size=segment_size)
        # Breadth, not provider depth: one in-flight job per provider/seat by default.
        self.per_provider_cap = max(1, int(per_provider_cap))
        self.per_seat_cap = max(1, int(per_seat_cap))
        self.max_attempts = max(1, int(max_attempts))
        self.poll_interval_seconds = max(5.0, float(poll_interval_seconds))
        self.materialize_concurrency = max(1, int(materialize_concurrency))
        self.fallback_concurrency = max(1, int(fallback_concurrency))
        self.submit_concurrency = max(1, int(submit_concurrency))
        self.execution_backend = execution_backend
        self.live_max_clips = int(max_clips or 0)
        if self.execution_backend == "live" and (not live or self.live_max_clips <= 0):
            raise ValueError("live video requires explicit live=True and a positive max_clips safety cap")

    async def aclose(self) -> None:
        await self.chat.aclose()
        await self.video.aclose()

    @staticmethod
    def _backoff(attempts: int) -> float:
        return min(15 * 60.0, 5.0 * (2 ** max(0, attempts - 1)))

    async def _configure_soundtrack(self, film: Dict[str, Any], *, enabled: bool = True) -> None:
        """Attach a durable Sound Director music bed to this film's final assembly.

        A short procedural score loops at mux time, so a 30-minute cut gets actual AAC
        audio without waiting for a 30-minute source render or relying on a third party.
        """
        self.assembler.narration = None
        if not enabled:
            self.assembler.music = None
            return
        audio_dir = self.assembler.root / str(film["id"]) / "audio"
        music = audio_dir / "sound_director_score.m4a"
        if not music.exists() or music.stat().st_size < 1024:
            duration = min(120.0, max(12.0, float(film.get("target_seconds") or 12)))
            try:
                await media.synthesize_music_bed(music, duration=duration)
            except Exception:
                # A valid audio stream is part of the deliverable even on an unusual
                # FFmpeg build that lacks a score filter used by the richer path.
                await media.synthesize_silence(music, duration=max(2.0, min(8.0, duration)))
        self.assembler.music = music

    async def request_film(self, request: FilmRequest) -> Dict[str, Any]:
        """Plan and persist an idempotent graph, recovering a zero-shot crash replay."""
        if not request.script.strip():
            raise ValueError("script must not be empty")
        target_seconds = max(5, int(request.target_seconds))
        if self.execution_backend == "live" and request.max_clips is None:
            raise ValueError("live film requests require a positive per-request max_clips cap")
        if self.execution_backend == "live" and int(request.max_clips or 0) > self.live_max_clips:
            raise ValueError("request max_clips exceeds this live orchestrator's explicit safety cap")

        stable_input_bible = request.bible or {"style": request.style, "imageRefs": request.image_refs}
        request_key = request_key_for_film(
            script=request.script, target_seconds=target_seconds, aspect_ratio=request.aspect_ratio,
            bible=stable_input_bible, seed=request.seed,
        )
        existing = self.store.get_film_by_request_key(request_key)
        if existing is not None and int(existing["logical_shots"]) > 0:
            await self._configure_soundtrack(existing, enabled=request.with_audio)
            return {"film": self.store.summary(existing["id"]), "idempotent_replay": True}

        # A zero-shot film is the old crash window between create_film and graph insert.
        bible = (existing or {}).get("bible") or request.bible
        if not bible:
            bible = await self.planner.create_bible(
                request.script, style=request.style, image_refs=request.image_refs
            )
        shots = await self.planner.decompose(
            request.script, target_seconds=target_seconds, bible=bible, seed=request.seed
        )
        canonical = {
            shot.canonical_index if shot.canonical_index is not None else shot.index: shot
            for shot in shots
        }
        required = len(canonical)

        capacity_error: Optional[str] = None
        try:
            capacity = await self.video.capacity()
            available = capacity.total_clips_available
        except VideoTransientError as exc:
            # A capacity endpoint outage must not discard a no-credit fallback film.
            available = 0
            capacity_error = str(exc)

        motion_limit = int(request.max_clips or 0) if self.execution_backend == "live" else 0
        if not request.allow_fallback and request.max_clips is not None and required > max(0, int(request.max_clips)):
            film = self.store.create_film(
                request_key=request_key, script=request.script, target_seconds=target_seconds,
                aspect_ratio=request.aspect_ratio, bible=bible, status="refused_safety_cap",
                admission_mode=request.admission_mode, capacity_at_admission=available,
                motion_clip_limit=motion_limit,
            )
            self.store.set_film_status(
                film["id"], "refused_safety_cap",
                error=f"motion safety cap is {int(request.max_clips)} clips but plan requires {required} unique clips",
            )
            return {
                "film": self.store.summary(film["id"]), "accepted": False,
                "required_unique_clips": required, "max_clips": int(request.max_clips),
                "shortfall": required - int(request.max_clips),
            }

        # Fallback makes lack of router capacity a quality decision, not a refusal.
        status = "admitted" if request.allow_fallback or available >= required else (
            "wave_scheduled" if request.admission_mode == "waves" else "refused_capacity"
        )
        film = self.store.create_film(
            request_key=request_key, script=request.script, target_seconds=target_seconds,
            aspect_ratio=request.aspect_ratio, bible=bible, status=status, admission_mode=request.admission_mode,
            capacity_at_admission=available, motion_clip_limit=motion_limit,
        )
        if int(film["logical_shots"]):
            await self._configure_soundtrack(film, enabled=request.with_audio)
            return {"film": self.store.summary(film["id"]), "idempotent_replay": True}

        shortfall = max(0, required - available)
        if status == "refused_capacity":
            self.store.set_film_status(
                film["id"], status,
                error=f"capacity shortfall: requires {required} unique clips; {available} available; shortfall {shortfall}",
            )
            return {
                "film": self.store.summary(film["id"]), "accepted": False, "required_unique_clips": required,
                "available_clips": available, "shortfall": shortfall,
            }

        job_keys = {index: job_key_for_shot(shot, canonical_index=index) for index, shot in canonical.items()}
        self.store.add_shots_and_jobs(film["id"], shots, aspect_ratio=request.aspect_ratio, job_keys=job_keys)
        self.store.set_film_status(film["id"], "admitted", error=None)
        film = self.store.get_film(film["id"]) or film
        await self._configure_soundtrack(film, enabled=request.with_audio)
        return {
            "film": self.store.summary(film["id"]), "accepted": True,
            "required_unique_clips": required, "logical_shots": len(shots),
            "available_clips": available, "shortfall": shortfall,
            "motion_clip_limit": motion_limit or None,
            "fallback_enabled": bool(request.allow_fallback),
            "capacity_strategy": "hybrid_fallback" if request.allow_fallback else request.admission_mode,
            **({"capacity_error": capacity_error} if capacity_error else {}),
        }

    def _choose_seat(
        self,
        capacity: Any,
        *,
        remaining: Dict[str, int],
        scheduled_provider: Dict[str, int],
        scheduled_seat: Dict[tuple[str, str], int],
    ) -> Optional[Any]:
        unavailable = self.store.unavailable_seats()
        choices = []
        for seat in capacity.seats:
            available = remaining.get(seat.seat, seat.clips_remaining)
            key = (seat.provider, seat.seat)
            if available <= 0 or key in unavailable:
                continue
            if self.store.provider_inflight(seat.provider) + scheduled_provider[seat.provider] >= self.per_provider_cap:
                continue
            if self.store.jobs_using_seat(seat.provider, seat.seat) + scheduled_seat[key] >= self.per_seat_cap:
                continue
            choices.append(seat)
        choices.sort(
            key=lambda s: (
                self.store.jobs_using_seat(s.provider, s.seat) + scheduled_seat[(s.provider, s.seat)],
                -remaining.get(s.seat, s.clips_remaining), s.provider, s.seat,
            )
        )
        return choices[0] if choices else None

    async def _materialize_completed(self, film_id: str, counts: Dict[str, int]) -> None:
        semaphore = asyncio.Semaphore(self.materialize_concurrency)

        async def one(job: Dict[str, Any]) -> None:
            async with semaphore:
                try:
                    await self.assembler.materialize_job(film_id, job, self.materializer)
                    counts["materialized"] += 1
                except Exception as exc:
                    self.store.defer(
                        job["job_key"], status="succeeded", seconds=0,
                        error=f"local materialization pending: {type(exc).__name__}: {exc}",
                    )
                    counts["materialization_errors"] += 1

        await asyncio.gather(*(one(job) for job in self.store.jobs_needing_materialization(film_id)))

    async def _materialize_fallbacks(self, film_id: str, counts: Dict[str, int]) -> None:
        """Render keyless MP4s in the exact canonical slots router clips would use."""
        semaphore = asyncio.Semaphore(self.fallback_concurrency)

        async def one(job: Dict[str, Any]) -> None:
            async with semaphore:
                shot = self.store.shot_for_job(film_id, str(job["job_key"]))
                if shot is None:
                    self.store.quarantine(str(job["job_key"]), "fallback has no logical shot")
                    counts["quarantined"] += 1
                    return
                destination = self.assembler.clip_path(film_id, str(job["job_key"]))
                try:
                    if await media.is_playable(destination):
                        self.store.mark_fallback_success(str(job["job_key"]), str(destination))
                    else:
                        path = await self.fallback_materializer.materialize_fallback(
                            prompt=str(shot["prompt"]), image_refs=list(shot["image_refs"]), seed=shot.get("seed"),
                            continuity=dict(shot.get("continuity") or {}), destination=destination,
                            duration_seconds=int(shot["duration_seconds"]), aspect_ratio=str(job["aspect_ratio"]),
                        )
                        if not await media.is_playable(path):
                            raise media.MediaError(f"fallback clip is not playable: {path}")
                        self.store.mark_fallback_success(str(job["job_key"]), str(path))
                    counts["fallback_materialized"] += 1
                except Exception as exc:
                    self.store.defer(
                        str(job["job_key"]), status="fallback_pending", seconds=15,
                        error=f"fallback materialization pending: {type(exc).__name__}: {exc}",
                    )
                    counts["fallback_errors"] += 1

        await asyncio.gather(*(one(job) for job in self.store.fallback_jobs(film_id)))

    def _fail_if_terminal_jobs(self, film_id: str, counts: Dict[str, int]) -> bool:
        failures = self.store.terminal_failures(film_id)
        if not failures:
            return False
        first = failures[0]
        self.store.set_film_status(
            film_id, "failed",
            error=f"terminal clip failure {first['job_key']}: {first.get('last_error') or first['status']}",
        )
        counts["failed"] += 1
        return True

    async def _submit_claimed(
        self,
        claimed: Dict[str, Any],
        *,
        allow_fallback: bool,
        counts: Dict[str, int],
    ) -> None:
        """POST only after the durable exclusive claim has committed."""
        key = str(claimed["job_key"])
        try:
            submitted = await self.video.submit(
                job_key=key, prompt=str(claimed["prompt"]), duration_seconds=int(claimed["duration_seconds"]),
                image_refs=list(claimed["image_refs"]), aspect_ratio=str(claimed["aspect_ratio"]),
                seed=claimed.get("seed"),
            )
            self.store.mark_submitted(
                key, remote_job_id=submitted.job_id, provider=submitted.provider,
                seat=claimed.get("seat"), credits_reserved=submitted.credits_reserved,
            )
            counts["submitted"] += 1
        except VideoNoCapacity as exc:
            if allow_fallback:
                self.store.mark_fallback_pending(key, f"capacity: {exc}")
                counts["fallback_capacity"] += 1
            else:
                self.store.defer(key, status="parked_capacity", seconds=exc.retry_after_seconds, error=str(exc))
                counts["parked_capacity"] += 1
        except VideoTransientError as exc:
            if allow_fallback and "safety cap" in str(exc).lower():
                self.store.mark_fallback_pending(key, f"motion budget: {exc}")
                counts["fallback_budget"] += 1
                return
            attempts = int(claimed["attempts"])
            if attempts >= self.max_attempts:
                self.store.quarantine(key, str(exc))
                counts["quarantined"] += 1
            else:
                self.store.defer(key, status="retry_wait", seconds=self._backoff(attempts), error=str(exc))
                counts["retrying"] += 1

    async def tick(self, film_id: str) -> Dict[str, Any]:
        """Advance one bounded, restart-safe scheduling pass for a film."""
        film = self.store.get_film(film_id)
        if not film:
            raise KeyError(f"film {film_id} does not exist")
        if film["status"] in self._TERMINAL_FILM_STATES:
            return {"film": self.store.summary(film_id), "actions": {"terminal": 1}}

        motion_limit = int(film.get("motion_clip_limit") or 0)
        if self.execution_backend == "live" and motion_limit <= 0:
            raise ValueError("refusing live tick: film has no persisted motion safety cap")
        counts: Dict[str, int] = defaultdict(int)
        counts["recovered_claims"] = self.store.recover_stale_claims(film_id)
        await self._configure_soundtrack(film, enabled=True)

        # Poll existing router work first. Store enforces the frozen >=5 s/job interval.
        for job in self.store.due_polls(film_id, poll_interval_seconds=self.poll_interval_seconds):
            try:
                remote = await self.video.get_job(str(job["remote_job_id"]))
                counts["polled"] += 1
                self.store.mark_poll(
                    job["job_key"], status=remote.status, provider=remote.provider, url=remote.url,
                    error=remote.error, credits_spent=remote.credits_spent,
                )
                if remote.status != "failed":
                    continue
                error_type = self.video.terminal_exception(remote.error)
                if isinstance(error_type, VideoWallError):
                    self.store.mark_fallback_pending(job["job_key"], f"wall: {remote.error or 'provider wall'}")
                    counts["fallback_wall"] += 1
                elif isinstance(error_type, VideoAuthError):
                    # The frozen contract has no seat hint/reassign operation for a
                    # stable failed jobKey. Park only this local seat and complete the
                    # slot locally; a true retry-on-another-seat needs that contract add.
                    self.store.mark_seat_unavailable(str(job.get("seat") or ""), remote.provider, remote.error or "")
                    self.store.mark_fallback_pending(job["job_key"], f"auth seat unavailable: {remote.error or 'unknown'}")
                    counts["fallback_auth"] += 1
                elif int(job["attempts"]) >= self.max_attempts:
                    self.store.quarantine(job["job_key"], remote.error or "repeated terminal failure")
                    counts["quarantined"] += 1
                else:
                    self.store.defer(
                        job["job_key"], status="retry_wait", seconds=self._backoff(int(job["attempts"])),
                        error=remote.error or "transient remote failure",
                    )
                    counts["retrying"] += 1
            except VideoTransientError as exc:
                # A failed GET says nothing about the remote render. Never spend an
                # attempt or blindly re-POST a still-running job.
                self.store.note_poll_transport_error(job["job_key"], str(exc))
                counts["poll_errors"] += 1

        await self._materialize_completed(film_id, counts)

        capacity = None
        try:
            capacity = await self.video.capacity()
        except VideoTransientError:
            # Capacity is advisory; completed and fallback materialisation can proceed.
            counts["capacity_errors"] += 1

        if capacity is not None:
            remaining = {seat.seat: seat.clips_remaining for seat in capacity.seats}
            scheduled_provider: Dict[str, int] = defaultdict(int)
            scheduled_seat: Dict[tuple[str, str], int] = defaultdict(int)
            remaining_motion = max(0, motion_limit - self.store.remote_submission_count(film_id)) if motion_limit else None
            claimed_jobs: List[Dict[str, Any]] = []

            # Claim all permitted work before awaiting any POST. This is the atomic
            # double-charge boundary and lets independent provider lanes submit together.
            for job in self.store.due_jobs(film_id, limit=10_000):
                if remaining_motion is not None and remaining_motion <= 0:
                    self.store.mark_fallback_pending(job["job_key"], "motion safety budget exhausted")
                    counts["fallback_budget"] += 1
                    continue
                seat = self._choose_seat(
                    capacity, remaining=remaining, scheduled_provider=scheduled_provider,
                    scheduled_seat=scheduled_seat,
                )
                if seat is None:
                    if any(value > 0 for value in remaining.values()):
                        counts["concurrency_limited"] += 1
                        break
                    self.store.mark_fallback_pending(job["job_key"], "capacity: no eligible video seat")
                    counts["fallback_capacity"] += 1
                    continue
                claimed = self.store.claim_for_submit(job["job_key"], provider=seat.provider, seat=seat.seat)
                if claimed is None:
                    continue
                remaining[seat.seat] = max(0, remaining.get(seat.seat, 0) - 1)
                scheduled_provider[seat.provider] += 1
                scheduled_seat[(seat.provider, seat.seat)] += 1
                if remaining_motion is not None:
                    remaining_motion -= 1
                claimed_jobs.append(claimed)
                if len(claimed_jobs) >= self.submit_concurrency:
                    break

            await asyncio.gather(
                *(self._submit_claimed(job, allow_fallback=True, counts=counts) for job in claimed_jobs)
            )

        await self._materialize_fallbacks(film_id, counts)
        await self._materialize_completed(film_id, counts)
        if self._fail_if_terminal_jobs(film_id, counts):
            return {"film": self.store.summary(film_id), "actions": dict(counts)}

        try:
            final = await self.assembler.assemble_available(film_id)
            if final:
                counts["finalized"] += 1
            elif self.store.get_film(film_id)["status"] not in self._TERMINAL_FILM_STATES:
                self.store.set_film_status(film_id, "running")
        except Exception as exc:
            self.store.set_film_status(film_id, "assembly_error", error=f"{type(exc).__name__}: {exc}")
            counts["assembly_errors"] += 1
        return {"film": self.store.summary(film_id), "actions": dict(counts)}

    async def run_until_idle(
        self,
        film_id: str,
        *,
        max_ticks: int = 100,
        sleep_seconds: float = 5.0,
    ) -> Dict[str, Any]:
        """Bounded operator/test helper; production calls ``tick`` from a worker."""
        latest: Dict[str, Any] = {}
        active = {"pending", "submitting", "queued", "running", "retry_wait", "parked_capacity", "fallback_pending"}
        for _ in range(max(1, int(max_ticks))):
            latest = await self.tick(film_id)
            states = latest["film"]["job_states"]
            if latest["film"]["status"] in self._TERMINAL_FILM_STATES:
                return latest
            if not any(state in states for state in active):
                return latest
            await asyncio.sleep(max(0.0, sleep_seconds))
        return latest
