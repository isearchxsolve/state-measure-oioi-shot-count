"""READ-ONLY measurement: how many shots does decompose emit for 30 minutes?

Imports oioi's film_pipeline.shots and runs it against the in-repo zero-network mock
chat client. Writes nothing into the oioi tree (PYTHONDONTWRITEBYTECODE is set by the
caller). Counts ordered shots and distinct canonical renders.
"""
import asyncio
import json
import os
import sys

sys.dont_write_bytecode = True
BACKEND = "D:/god_ai/oioi/oioi/backend"
sys.path.insert(0, BACKEND)

from film_pipeline.shots import ShotListPlanner  # noqa: E402
from film_pipeline.mock import DeterministicMockChatClient  # noqa: E402


class BrokenChat:
    """Forces the local storyboard fallback path (what a real 503/garbage reply hits)."""

    async def aclose(self):
        return None

    async def message(self, content, *, max_tokens=1024, model="pool/auto"):
        raise RuntimeError("chat pool unavailable (simulated)")


async def main():
    out = {}
    bible = {
        "characters": {"lead": "protagonist"}, "wardrobe": "coat", "locations": ["dock"],
        "lighting": "dusk", "style": "cinematic", "imageRefs": [],
    }
    for label, chat in (("mock_chat", DeterministicMockChatClient()), ("local_fallback", BrokenChat())):
        planner = ShotListPlanner(chat)
        row = {"batch_size": planner.batch_size}
        for name, seconds in (("30min_1800s", 1800), ("3h_10800s", 10800), ("60s", 60)):
            shots = await planner.decompose(
                "INT. DOCK - DUSK\nThe lead waits.\n", target_seconds=seconds,
                bible=bible, seed=20260826,
            )
            canonical = {
                (s.canonical_index if s.canonical_index is not None else s.index) for s in shots
            }
            row[name] = {
                "ordered_shots": len(shots),
                "distinct_canonical_renders": len(canonical),
                "max_index": max(s.index for s in shots),
                "durations_all_5s": all(s.duration_seconds == 5 for s in shots),
                "chat_batches_issued": -(-len(shots) // planner.batch_size),
            }
        out[label] = row
    print(json.dumps(out, indent=2))


asyncio.run(main())
