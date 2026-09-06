"""READ-ONLY measurement: does a 503 no_capacity / wall: error yield a Ken Burns clip
in the SAME ordered assembly slot?

Scenario C: zero seats -> every POST /v1/video/jobs answers 503 {"error":"no_capacity"}.
Scenario D: seats exist but every job returns a terminal error prefixed "wall:".
Asserts on the store: are all N ordered shot slots filled, and by what.
"""
import asyncio
import json
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, "D:/god_ai/oioi/oioi/backend")

import httpx  # noqa: E402

from film_pipeline.assembly import SyntheticClipMaterializer, SyntheticFallbackClipMaterializer  # noqa: E402
from film_pipeline.clients import VideoRouterClient  # noqa: E402
from film_pipeline.mock import MockSeat, MockVideoBackend, MockVideoConfig  # noqa: E402
from film_pipeline.orchestrator import FilmOrchestrator, FilmRequest  # noqa: E402
from film_pipeline.store import FilmJobStore  # noqa: E402
from measure_serial import DistinctChat  # noqa: E402

SCRATCH = Path("D:/omniroute-monetizer/state/measure/oioi-shot-count/run")


async def scenario(label, *, seats, terminal_failures, target_seconds, ticks):
    root = SCRATCH / label
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    backend = MockVideoBackend(MockVideoConfig(seats=seats, terminal_failures=terminal_failures))
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=backend.app), base_url="http://mock-video")
    store = FilmJobStore(root / "jobs.sqlite3")
    orch = FilmOrchestrator(
        store=store, chat=DistinctChat(), video=VideoRouterClient(client=http),
        artifact_root=root / "artifacts", materializer=SyntheticClipMaterializer(),
        fallback_materializer=SyntheticFallbackClipMaterializer(), segment_size=4,
    )
    admitted = await orch.request_film(FilmRequest(
        script="INT. DOCK - DUSK\nThe lead waits.\n", target_seconds=target_seconds,
        with_audio=False, seed=11,
    ))
    film_id = admitted["film"]["id"]
    totals = {}
    for _ in range(ticks):
        res = await orch.tick(film_id)
        for k, v in res["actions"].items():
            totals[k] = totals.get(k, 0) + v
        if res["film"]["status"] in {"completed", "failed", "assembly_error"}:
            break

    shots = store.shots(film_id)
    jobs = {j["job_key"]: j for j in store.jobs_for_film(film_id)}
    filled, by_kind, gaps = 0, {}, []
    for s in shots:
        j = jobs.get(s["job_key"])
        ok = bool(j and j["status"] == "succeeded" and j.get("local_path"))
        if ok:
            filled += 1
            kind = "fallback(kenburns_slot)" if j.get("is_fallback") else "router_motion"
            by_kind[kind] = by_kind.get(kind, 0) + 1
        else:
            gaps.append({"shot_index": s["shot_index"], "status": j["status"] if j else "no-job"})
    summary = store.summary(film_id)
    reasons = sorted({str(j.get("fallback_reason"))[:60] for j in jobs.values() if j.get("fallback_reason")})
    segs = store.segments(film_id)
    await http.aclose()
    return {
        "admitted_status": admitted["film"]["status"],
        "final_film_status": summary["status"],
        "ordered_shot_slots": len(shots),
        "slots_filled": filled,
        "slots_by_source": by_kind,
        "unfilled_slots": gaps[:5],
        "fallback_reasons_seen": reasons,
        "segments_built": len(segs),
        "segment_index_ranges": [(s["start_shot_index"], s["end_shot_index"]) for s in segs],
        "final_mp4_exists": bool(summary.get("final_path")) and Path(str(summary["final_path"])).exists(),
        "action_totals": totals,
    }


async def main():
    out = {}
    out["C_zero_seats_503_no_capacity"] = await scenario(
        "c_no_capacity", seats=[], terminal_failures={}, target_seconds=60, ticks=6,
    )
    # Every submitted job comes back terminally failed with a "wall:" prefix.
    backend_probe = MockVideoBackend(MockVideoConfig())
    out["D_wall_error"] = await scenario(
        "d_wall", seats=[MockSeat(seat="w1", provider="p1", clips_remaining=99)],
        terminal_failures={"*": "wall: credit wall reached"}, target_seconds=60, ticks=12,
    )
    print(json.dumps(out, indent=2, default=str))


asyncio.run(main())
