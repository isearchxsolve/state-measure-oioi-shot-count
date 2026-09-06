"""READ-ONLY measurement: is film_pipeline submission serial or parallel?

Runs oioi's real FilmOrchestrator against its own in-repo MockVideoBackend, configured
to mirror the REAL /v1/video/capacity shape measured from :8189 (4 seats, 1 provider),
and then again with 4 DISTINCT providers.  Counts clips submitted per tick and the peak
observed overlap inside submit().  Writes only under this scratch directory.
"""
import asyncio
import json
import shutil
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, "D:/god_ai/oioi/oioi/backend")

import httpx  # noqa: E402

from film_pipeline.assembly import SyntheticClipMaterializer, SyntheticFallbackClipMaterializer  # noqa: E402
from film_pipeline.clients import VideoRouterClient  # noqa: E402
from film_pipeline.mock import MockSeat, MockVideoBackend, MockVideoConfig  # noqa: E402
from film_pipeline.models import ChatReply  # noqa: E402
from film_pipeline.orchestrator import FilmOrchestrator, FilmRequest  # noqa: E402
from film_pipeline.store import FilmJobStore  # noqa: E402

SCRATCH = Path("D:/omniroute-monetizer/state/measure/oioi-shot-count/run")


class DistinctChat:
    """Every beat is a distinct render payload, as a real chat pool would produce."""

    async def aclose(self):
        return None

    async def message(self, content, *, max_tokens=1024, model="pool/auto"):
        if "continuity bible" in content.lower():
            return ChatReply(text=json.dumps({
                "characters": {"lead": "protagonist"}, "wardrobe": "coat",
                "locations": ["dock"], "lighting": "dusk", "style": "cinematic", "imageRefs": [],
            }), provider="probe-chat")
        import re
        m = re.search(r"BATCH:\s*shots\s+(\d+)\s+through\s+(\d+)", content, re.I)
        first, last = int(m.group(1)), int(m.group(2))
        return ChatReply(text=json.dumps([
            {"prompt": f"distinct ordered beat {i} of the film", "imageRefs": [],
             "continuity": {"motion": "slow push in"}}
            for i in range(first, last + 1)
        ]), provider="probe-chat")


class OverlapProbe(VideoRouterClient):
    """Records concurrency actually achieved across submit() calls."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.inflight = 0
        self.peak_inflight = 0
        self.submit_windows = []

    async def submit(self, **kw):
        self.inflight += 1
        self.peak_inflight = max(self.peak_inflight, self.inflight)
        t0 = time.perf_counter()
        try:
            # Hold the window open so genuinely parallel submits must overlap.
            await asyncio.sleep(0.25)
            return await super().submit(**kw)
        finally:
            self.submit_windows.append((t0, time.perf_counter()))
            self.inflight -= 1


def peak_overlap(windows):
    events = []
    for start, end in windows:
        events.append((start, 1))
        events.append((end, -1))
    events.sort()
    cur = peak = 0
    for _, delta in events:
        cur += delta
        peak = max(peak, cur)
    return peak


async def scenario(label, seats, target_seconds, ticks=3):
    root = SCRATCH / label
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    backend = MockVideoBackend(MockVideoConfig(seats=seats))
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=backend.app), base_url="http://mock-video")
    video = OverlapProbe(client=http)
    orch = FilmOrchestrator(
        store=FilmJobStore(root / "jobs.sqlite3"), chat=DistinctChat(), video=video,
        artifact_root=root / "artifacts", materializer=SyntheticClipMaterializer(),
        fallback_materializer=SyntheticFallbackClipMaterializer(), segment_size=2,
    )
    admitted = await orch.request_film(FilmRequest(
        script="INT. DOCK - DUSK\nThe lead waits for the boat.\n",
        target_seconds=target_seconds, with_audio=False, seed=7,
    ))
    film_id = admitted["film"]["id"]
    out = {
        "orchestrator_defaults": {
            "submit_concurrency": orch.submit_concurrency,
            "per_provider_cap": orch.per_provider_cap,
            "per_seat_cap": orch.per_seat_cap,
            "poll_interval_seconds": orch.poll_interval_seconds,
        },
        "capacity_shape": {
            "seat_count": len(seats),
            "distinct_providers": len({s.provider for s in seats}),
        },
        "logical_shots": admitted.get("logical_shots"),
        "required_unique_clips": admitted.get("required_unique_clips"),
        "ticks": [],
    }
    for n in range(ticks):
        before = len(video.submit_windows)
        result = await orch.tick(film_id)
        acts = result["actions"]
        window_slice = video.submit_windows[before:]
        out["ticks"].append({
            "tick": n + 1,
            "submitted": acts.get("submitted", 0),
            "concurrency_limited": acts.get("concurrency_limited", 0),
            "fallback_capacity": acts.get("fallback_capacity", 0),
            "polled": acts.get("polled", 0),
            "peak_overlap_this_tick": peak_overlap(window_slice),
        })
    out["peak_inflight_submits_overall"] = video.peak_inflight
    await http.aclose()
    return out


async def main():
    real_shape = [
        MockSeat(seat="s1", provider="rtst.ai", clips_remaining=1),
        MockSeat(seat="s2", provider="rtst.ai", clips_remaining=5),
        MockSeat(seat="s3", provider="rtst.ai", clips_remaining=5),
        MockSeat(seat="s4", provider="rtst.ai", clips_remaining=5),
    ]
    multi_provider = [
        MockSeat(seat="s1", provider="p1", clips_remaining=5),
        MockSeat(seat="s2", provider="p2", clips_remaining=5),
        MockSeat(seat="s3", provider="p3", clips_remaining=5),
        MockSeat(seat="s4", provider="p4", clips_remaining=5),
    ]
    results = {
        "A_real_8189_shape_1_provider_4_seats": await scenario("real_shape", real_shape, 100),
        "B_hypothetical_4_providers": await scenario("multi_provider", multi_provider, 100),
    }
    print(json.dumps(results, indent=2))


asyncio.run(main())
