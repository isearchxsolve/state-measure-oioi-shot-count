"""Incremental ordered assembly for durable clip-job output."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from studio_pipeline import media

from .store import FilmJobStore


class ClipMaterializer(Protocol):
    async def materialize(self, *, remote_url: str, destination: Path, duration_seconds: int, aspect_ratio: str) -> Path:
        """Get a completed router clip into the durable local film storage."""


class FallbackClipMaterializer(Protocol):
    """Materialise an ordered shot when the async video pool cannot supply motion.

    The fallback deliberately produces the same uniform local MP4 contract as a router
    clip.  Assembly therefore keeps the original shot order and never has to special-case
    a provider outage, exhausted free seats, or a terminal provider wall.
    """

    async def materialize_fallback(
        self,
        *,
        prompt: str,
        image_refs: List[str],
        seed: Optional[int],
        continuity: Dict[str, Any],
        destination: Path,
        duration_seconds: int,
        aspect_ratio: str,
    ) -> Path:
        """Create a playable fallback MP4 in ``destination``."""


class HttpClipMaterializer:
    """Download, normalise, and atomically retain a real router URL for assembly."""

    async def materialize(self, *, remote_url: str, destination: Path, duration_seconds: int, aspect_ratio: str) -> Path:
        if remote_url.startswith("mock://"):
            raise media.MediaError("mock URL needs SyntheticClipMaterializer; no remote URL was fetched")
        source = destination.with_suffix(".source" + (Path(remote_url).suffix or ".mp4"))
        await media.download(remote_url, source)
        width, height = media.resolve_size(aspect_ratio)
        await media.normalize_clip(source, destination, duration=float(duration_seconds), width=width, height=height)
        return destination


class SyntheticClipMaterializer:
    """Generate a valid zero-credit slate clip for mock-only end-to-end tests."""

    def __init__(self, *, width: int = 320, height: int = 180) -> None:
        self.width = max(16, width)
        self.height = max(16, height)

    async def materialize(self, *, remote_url: str, destination: Path, duration_seconds: int, aspect_ratio: str) -> Path:
        # ``remote_url`` is intentionally not fetched: mock:// denotes the local test backend.
        await media.synthesize_slate_clip(
            destination, duration=max(0.5, float(duration_seconds)), width=self.width, height=self.height
        )
        return destination


class KenBurnsFallbackMaterializer:
    """Keyless still-image to motion fallback for every unavailable router clip.

    The primary fallback is a Pollinations still followed by the existing FFmpeg Ken
    Burns synthesiser.  A caller supplied image reference is preferred when available so
    character sheets and approved scene art carry through to the fallback.  If a still
    provider is temporarily unavailable, the final slate tier preserves the exact edit
    slot and duration rather than leaving the film permanently incomplete.
    """

    def __init__(self, *, width: Optional[int] = None, height: Optional[int] = None) -> None:
        self.width = width
        self.height = height

    @staticmethod
    def _motion(continuity: Dict[str, Any]) -> str:
        shot = continuity.get("shot") if isinstance(continuity, dict) else None
        if not isinstance(shot, dict):
            return "slow push in"
        for key in ("motion", "camera", "camera_move", "movement"):
            value = shot.get(key)
            if isinstance(value, str) and value.strip():
                return value
            if isinstance(value, dict):
                for nested in ("movement", "move", "motion"):
                    candidate = value.get(nested)
                    if isinstance(candidate, str) and candidate.strip():
                        return candidate
        return "slow push in"

    @staticmethod
    def _reference_url(image_refs: List[str]) -> Optional[str]:
        for ref in image_refs:
            value = str(ref or "").strip()
            if value.startswith(("https://", "http://")):
                return value
        return None

    async def materialize_fallback(
        self,
        *,
        prompt: str,
        image_refs: List[str],
        seed: Optional[int],
        continuity: Dict[str, Any],
        destination: Path,
        duration_seconds: int,
        aspect_ratio: str,
    ) -> Path:
        width, height = media.resolve_size(aspect_ratio)
        width = self.width or width
        height = self.height or height
        # The still is durable too: resume never asks the image service for the same shot
        # twice after a worker interruption.
        still = destination.parent.parent / "stills" / f"{destination.stem}.jpg"
        try:
            if not still.exists() or still.stat().st_size < 1024:
                source = self._reference_url(image_refs) or media.pollinations_image_url(
                    prompt, width=width, height=height, seed=seed
                )
                await media.download(source, still, timeout=240.0, attempts=3, min_bytes=1024)
            await media.synthesize_kenburns_clip(
                still,
                destination,
                duration=max(1.0, float(duration_seconds)),
                width=width,
                height=height,
                motion=self._motion(continuity),
            )
            return destination
        except Exception:
            # A provider outage must not turn zero available motion seats into an
            # unfinishable film.  The slate is intentionally conspicuous but it is a real,
            # correctly timed MP4 and therefore keeps every downstream edit slot aligned.
            await media.synthesize_slate_clip(
                destination,
                duration=max(1.0, float(duration_seconds)),
                width=width,
                height=height,
            )
            return destination


class SyntheticFallbackClipMaterializer:
    """Network-free fallback used by stub and unit end-to-end tests."""

    def __init__(self, *, width: int = 320, height: int = 180) -> None:
        self.width = max(16, width)
        self.height = max(16, height)

    async def materialize_fallback(
        self,
        *,
        prompt: str,
        image_refs: List[str],
        seed: Optional[int],
        continuity: Dict[str, Any],
        destination: Path,
        duration_seconds: int,
        aspect_ratio: str,
    ) -> Path:
        await media.synthesize_slate_clip(
            destination,
            duration=max(0.5, float(duration_seconds)),
            width=self.width,
            height=self.height,
        )
        return destination


class IncrementalAssembler:
    """Build fixed-size contiguous segments, then one final container at completion.

    A newly completed clip is never appended by re-encoding the previous entire film.
    At most one fixed-size segment is encoded per assembly event, which gives a
    2,160-shot film bounded incremental work and recovery checkpoints.
    """

    def __init__(
        self,
        store: FilmJobStore,
        artifact_root: Path | str,
        *,
        segment_size: int = 60,
        transition: str = "hard-cut",
        narration: Optional[Path] = None,
        music: Optional[Path] = None,
    ) -> None:
        self.store = store
        self.root = Path(artifact_root).resolve()
        self.segment_size = max(1, int(segment_size))
        self.transition = transition
        self.narration = narration
        self.music = music

    def clip_path(self, film_id: str, job_key: str) -> Path:
        return self.root / film_id / "clips" / f"{job_key}.mp4"

    async def materialize_job(self, film_id: str, job: Dict[str, object], materializer: ClipMaterializer) -> Path:
        existing = job.get("local_path")
        if existing and await media.is_playable(Path(str(existing))):
            return Path(str(existing))
        remote_url = str(job.get("remote_url") or "")
        if not remote_url:
            raise media.MediaError(f"succeeded job {job['job_key']} has no video URL")
        destination = self.clip_path(film_id, str(job["job_key"]))
        if await media.is_playable(destination):
            self.store.mark_local_success(str(job["job_key"]), str(destination))
            return destination
        path = await materializer.materialize(
            remote_url=remote_url, destination=destination, duration_seconds=int(job["duration_seconds"]),
            aspect_ratio=str(job["aspect_ratio"]),
        )
        if not await media.is_playable(path):
            raise media.MediaError(f"materialized clip is not playable: {path}")
        self.store.mark_local_success(str(job["job_key"]), str(path))
        return path

    async def assemble_available(self, film_id: str) -> Optional[Path]:
        """Emit next completed contiguous segment(s); finalise only once all shots resolve."""
        shots = self.store.shots(film_id)
        if not shots:
            return None
        jobs = {job["job_key"]: job for job in self.store.jobs_for_film(film_id)}
        segments = self.store.segments(film_id)
        start = int(segments[-1]["end_shot_index"]) + 1 if segments else 0
        segment_index = len(segments)
        film_dir = self.root / film_id
        segment_dir = film_dir / "segments"
        segment_dir.mkdir(parents=True, exist_ok=True)

        while start < len(shots):
            contiguous = []
            for shot in shots[start:]:
                job = jobs.get(shot["job_key"])
                if not job or job["status"] != "succeeded" or not job.get("local_path"):
                    break
                path = Path(str(job["local_path"]))
                if not await media.is_playable(path):
                    break
                contiguous.append(path)
                if len(contiguous) >= self.segment_size:
                    break
            remaining_total = len(shots) - start
            if len(contiguous) < self.segment_size and len(contiguous) != remaining_total:
                break  # preserve ordered playback; wait for the gap, do not build a fake cut
            if not contiguous:
                break
            end = start + len(contiguous) - 1
            destination = segment_dir / f"segment_{segment_index:05d}.mp4"
            manifest = segment_dir / f"segment_{segment_index:05d}.txt"
            if not await media.is_playable(destination):
                # hard-cut is a deliberate, stable transition profile.  Crossfades are optional
                # later because they force re-encoding at segment boundaries.
                await media.concat_copy(contiguous, destination, manifest=manifest)
            self.store.save_segment(
                film_id, segment_index=segment_index, start_shot_index=start, end_shot_index=end,
                path=str(destination), status=f"succeeded:{self.transition}",
            )
            start = end + 1
            segment_index += 1

        segments = self.store.segments(film_id)
        if not segments or int(segments[-1]["end_shot_index"]) != len(shots) - 1:
            return None
        final_dir = film_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        raw = final_dir / "assembled.mp4"
        final = final_dir / "film.mp4"
        segment_paths = [Path(str(segment["path"])) for segment in segments]
        if not await media.is_playable(raw):
            await media.concat_copy(segment_paths, raw, manifest=final_dir / "final_concat.txt")
        if not await media.is_playable(final):
            await media.mix_final_audio(raw, final, narration=self.narration, music=self.music, loop_music=True)
        self.store.set_film_status(film_id, "completed", output_path=str(final))
        return final
