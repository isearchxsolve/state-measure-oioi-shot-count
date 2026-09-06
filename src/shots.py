"""Deterministic, cached screenplay-to-shot decomposition and shot deduplication."""
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Dict, Iterable, List, Optional

from .clients import ChatPoolClient
from .models import ShotSpec


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def job_key_for_shot(shot: ShotSpec, *, canonical_index: int) -> str:
    """Caller-side stable idempotency key required by the frozen video contract.

    The index is the canonical/source shot index.  Repeated shots reference the same
    canonical index so one render is reusable, while a materially distinct shot at a
    later position cannot accidentally collide with it.
    """
    return stable_hash({
        "prompt": normalize_prompt(shot.prompt), "duration": int(shot.duration_seconds),
        "imageRefs": sorted(shot.image_refs), "seed": shot.seed, "shotIndex": int(canonical_index),
    })


def request_key_for_film(
    *, script: str, target_seconds: int, aspect_ratio: str, bible: Dict[str, Any], seed: Optional[int]
) -> str:
    return stable_hash({
        "script": script.strip(), "targetSeconds": int(target_seconds), "aspectRatio": aspect_ratio,
        "bible": bible, "seed": seed,
    })


_WORD = re.compile(r"[a-z0-9]+")


def normalize_prompt(prompt: str) -> str:
    """Stable prompt text for a *job key*, without using it for creative deduplication."""
    return " ".join(_WORD.findall((prompt or "").lower()))


def canonical_render_payload(shot: ShotSpec) -> Dict[str, Any]:
    """The only safe reuse boundary for a billable/fallback render.

    Earlier code used unordered word-overlap and equal position-derived seeds.  That made
    default dedup inert while also allowing ``A hits B`` and ``B hits A`` to collide when
    no seed was supplied.  A repeated cut may reuse only byte-identical creative render
    inputs.  The seed is intentionally excluded: the planner's position-derived seed is
    not creative intent and the canonical source shot owns the reused render seed.
    """
    return {
        "prompt": str(shot.prompt).strip(),
        "durationSeconds": int(shot.duration_seconds),
        # Reference order can change a multimodal provider result, so do not sort it.
        "imageRefs": [str(value) for value in shot.image_refs],
    }


def deduplicate_shots(shots: Iterable[ShotSpec]) -> List[ShotSpec]:
    """Reference only exact canonical render payloads; never fuzzy-match film actions."""
    result: List[ShotSpec] = []
    canonical: Dict[str, ShotSpec] = {}
    for shot in shots:
        payload_key = canonical_json(canonical_render_payload(shot))
        prior = canonical.get(payload_key)
        if prior is None:
            prior = shot
            canonical[payload_key] = prior
        # Identical prompts at different timeline positions deliberately inherit the
        # first render's seed/job.  A new seed remains a new render only when the caller
        # makes the visual payload distinct (for example a prompt variant/reference).
        result.append(ShotSpec(
            index=shot.index,
            prompt=shot.prompt,
            duration_seconds=shot.duration_seconds,
            image_refs=list(shot.image_refs),
            seed=prior.seed,
            continuity=dict(shot.continuity),
            hero_shot=bool(shot.hero_shot),
            canonical_index=prior.index,
        ))
    return result


def _extract_json(text: str) -> Any:
    clean = text.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", clean, flags=re.IGNORECASE)
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        # Providers regularly add one sentence before/after valid JSON, and an answer
        # can include a schema example before the requested payload.  Taking the first
        # opener and *last* closer joins two valid JSON values into invalid text.  Decode
        # each possible opener and accept the first complete structured value instead.
        decoder = json.JSONDecoder()
        for start, character in enumerate(clean):
            if character not in "[{":
                continue
            try:
                value, _ = decoder.raw_decode(clean[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, (dict, list)):
                return value
        raise ValueError("chat response did not contain a complete JSON object or array")


class ShotListPlanner:
    """Uses the chat pool in batched calls while retaining a compact continuity bible."""

    def __init__(self, chat: ChatPoolClient, *, batch_size: int = 40) -> None:
        self.chat = chat
        self.batch_size = max(1, int(batch_size))

    @staticmethod
    def _local_storyboard_batch(
        *,
        script: str,
        bible: Dict[str, Any],
        start: int,
        count: int,
        total: int,
        reason: str,
    ) -> List[Dict[str, Any]]:
        """Produce a usable, stable shot list if an advisory chat reply is malformed.

        The chat pool is valuable direction, but it must not be the single point that
        prevents a local 30-minute render.  This deliberately does *not* pretend a
        malformed response was approved creative work: every slot carries its fallback
        provenance in continuity.  It does guarantee the requested count, distinct
        ordered render payloads, the persistent bible, and three Director hero beats.
        """
        compact_script = " ".join(str(script).split())[:900]
        characters = bible.get("characters") or {}
        cast = ", ".join(str(key) for key in characters) if isinstance(characters, dict) else str(characters)
        locations = bible.get("locations") or []
        location = str(locations[0]) if locations else "the established story world"
        style = str(bible.get("style") or "cinematic")
        wardrobe = str(bible.get("wardrobe") or "consistent wardrobe")
        lighting = str(bible.get("lighting") or "continuous cinematic lighting")
        camera_moves = ("slow push in", "measured lateral track", "wide establishing frame", "intimate over-shoulder", "gentle crane reveal")
        beats: List[Dict[str, Any]] = []
        hero_indices = {0, max(0, total // 2), max(0, total - 1)}
        for offset in range(count):
            index = start + offset
            move = camera_moves[index % len(camera_moves)]
            beats.append({
                "prompt": (
                    f"{style} film, ordered storyboard beat {index + 1} of {total}; "
                    f"{cast or 'the established protagonist'} in {location}, {wardrobe}, {lighting}; "
                    f"{move}; story context: {compact_script}"
                ),
                "imageRefs": list(bible.get("imageRefs") or []),
                "continuity": {
                    "motion": move,
                    "plannerFallback": True,
                    "plannerFallbackReason": reason[:240],
                    "storyBeat": index + 1,
                },
                "heroShot": index in hero_indices,
            })
        return beats

    async def create_bible(self, script: str, *, style: str = "cinematic", image_refs: Optional[List[str]] = None) -> Dict[str, Any]:
        prompt = (
            "Create a compact film continuity bible from this screenplay. Return JSON only with "
            "keys characters (object of stable visual descriptions), wardrobe, locations, lighting, "
            "style, and imageRefs (array). Do not add prose outside JSON.\n\nSCREENPLAY:\n" + script
        )
        reply = await self.chat.message(prompt, max_tokens=1024, model="pool/auto")
        data = _extract_json(reply.text)
        if not isinstance(data, dict):
            raise ValueError("continuity bible must be a JSON object")
        data.setdefault("style", style)
        data.setdefault("characters", {})
        data.setdefault("wardrobe", "consistent with character bible")
        data.setdefault("locations", [])
        data.setdefault("lighting", "continuous cinematic lighting")
        data["imageRefs"] = list(image_refs or data.get("imageRefs") or [])
        return data

    async def decompose(
        self,
        script: str,
        *,
        target_seconds: int,
        bible: Dict[str, Any],
        seed: Optional[int],
    ) -> List[ShotSpec]:
        """Make batched free chat calls (not one call per five-second clip)."""
        total = max(1, int(math.ceil(int(target_seconds) / 5)))
        output: List[ShotSpec] = []
        compact_bible = canonical_json(bible)
        for start in range(0, total, self.batch_size):
            count = min(self.batch_size, total - start)
            request = (
                "Turn the screenplay into the next ordered five-second film beats. Return JSON only: "
                "an array of exactly {count} objects with prompt, imageRefs, continuity, and heroShot keys. "
                "Every prompt must preserve this persistent bible's characters, wardrobe, location and lighting. "
                "Avoid introducing a new character unless the screenplay requires it. Use the image refs supplied "
                "in the bible. Do not number or wrap the JSON.\n\n"
                "BIBLE:\n{bible}\n\nSCREENPLAY:\n{script}\n\nBATCH: shots {first} through {last}."
            ).format(count=count, bible=compact_bible, script=script, first=start + 1, last=start + count)
            provider = "local-storyboard-fallback"
            try:
                reply = await self.chat.message(request, max_tokens=max(1024, count * 80), model="pool/auto")
                provider = reply.provider
                raw = _extract_json(reply.text)
                if isinstance(raw, dict):
                    raw = raw.get("shots")
                if not isinstance(raw, list) or len(raw) != count:
                    received = len(raw) if isinstance(raw, list) else "non-list"
                    raw = self._local_storyboard_batch(
                        script=script, bible=bible, start=start, count=count, total=total,
                        reason=f"chat batch {start // self.batch_size + 1} returned {received}; expected {count}",
                    )
            except Exception as exc:
                raw = self._local_storyboard_batch(
                    script=script, bible=bible, start=start, count=count, total=total,
                    reason=f"chat batch {start // self.batch_size + 1} failed: {type(exc).__name__}: {exc}",
                )
            for offset, item in enumerate(raw):
                if not isinstance(item, dict) or not str(item.get("prompt") or "").strip():
                    raise ValueError(f"shot {start + offset + 1} has no generation prompt")
                refs = item.get("imageRefs", bible.get("imageRefs") or [])
                if not isinstance(refs, list):
                    refs = list(bible.get("imageRefs") or [])
                output.append(ShotSpec(
                    index=start + offset,
                    prompt=str(item["prompt"]).strip(), duration_seconds=5,
                    image_refs=[str(value) for value in refs],
                    seed=(int(seed) + start + offset) if seed is not None else None,
                    continuity={
                        "bible": bible,
                        "shot": item.get("continuity") or {},
                        "chat_provider": provider,
                    },
                    hero_shot=bool(item.get("heroShot", False)),
                ))
        return deduplicate_shots(output)
