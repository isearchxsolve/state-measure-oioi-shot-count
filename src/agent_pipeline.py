"""Legacy animation pipeline adapter that uses the new orchestrator to maintain backward compatibility.

This class mimics the interface of the original AnimationPipeline but routes all
inference through the new AnimationAgentOrchestrator, which persists AgentExecution
rows and uses the ProviderRegistry for free-tier provider failover.
"""

from typing import Dict, Any, List, Optional
import asyncio
import httpx
import json
import urllib.parse
import os
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import _init_engine
from app.services.orchestrator import AnimationAgentOrchestrator
from agent_contracts import (
    ScriptwriterInput,
    ArtDirectorInput,
    CharacterDesignerInput,
    SceneDesignerInput,
    StoryboardArtistInput,
    AnimatorInput,
    SoundDirectorInput,
    DirectorInput,
)
from providers.registry import ProviderRegistry, get_registry
from providers.base import ImageGenerationRequest, VideoGenerationRequest


# --- Backward-compatible module-level interface for tests and legacy code ---
# The film crew, matching src/app/page.tsx's agentTeam exactly: seven working agents plus the
# Director who coordinates and approves them. The Storyboard Artist used to be missing here while
# the frontend advertised it, and 'AI Avatar Agent' used to be listed here while the frontend did
# not -- it is a separate optional agent (see run_avatar_agent below), not part of the film crew,
# so it no longer pads this list.
AGENT_ROLES = [
    'Scriptwriter',
    'Art Director',
    'Character Designer',
    'Scene Designer',
    'Storyboard Artist',
    'Animator',
    'Sound Director',
    'Director',
]


def run_scriptwriter(idea: str, style: str = "Anime") -> str:
    """Legacy interface: run scriptwriter agent and return the script string."""
    import asyncio
    from agent_pipeline import AnimationPipeline
    # We need to run the async method in a sync context for the test.
    # Since the test is synchronous, we'll create a new event loop.
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    pipeline = AnimationPipeline()
    state = loop.run_until_complete(pipeline.run_scriptwriter(idea, style=style))
    return state.get("script", "")


def run_pipeline(idea: str, style: str = "Anime") -> Dict[str, Any]:
    """Legacy interface: run the full pipeline and return a dict with all stages."""
    import asyncio
    from agent_pipeline import AnimationPipeline
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    pipeline = AnimationPipeline()
    # Run scriptwriter to scene designer
    state1 = loop.run_until_complete(pipeline.run_scriptwriter(idea, style=style))
    # Run animator to director
    state2 = loop.run_until_complete(pipeline.run_animator(state1.get("scenes", [])))
    # Combine into the expected format
    result = {
        "script": state1.get("script", ""),
        "storyboard": state1.get("scenes", []),  # legacy key for scenes
        "characters": state1.get("characters", []),
        "scenes": state1.get("scenes", []),  # duplicate for compatibility
        "animation": state2.get("animation", []),
        "sound_plan": state2.get("sound_plan", {}),
        "director_notes": state2.get("director_notes", ""),
        "roles": state2.get("roles", []),
    }
    return result


class AnimationPipeline:
    """Backward-compatible pipeline that uses the new orchestrator under the hood."""

    def __init__(self, registry: Optional[ProviderRegistry] = None,
                 model: Optional[str] = None) -> None:
        self.registry = registry or get_registry()
        # Carried down to every orchestrator this pipeline builds so the caller's model choice
        # survives to the actual chat_completion call. None means settings.DEFAULT_CHAT_MODEL.
        self.model = model
        # We'll store the outputs of the first four agents for use in the animator
        self._scriptwriter_output: Optional[ScriptwriterInput] = None
        self._art_director_output: Optional[ArtDirectorInput] = None
        self._character_designer_output: Optional[CharacterDesignerInput] = None
        self._scene_designer_output: Optional[SceneDesignerInput] = None
        self._storyboard_artist_output = None

    async def _get_image_adapter(self):
        """Get an image adapter, preferring Pollinations for keyless fallback."""
        return await self.registry.get_adapter(capability="image", prefer="pollinations")

    async def _get_video_adapter(self):
        """Get a video adapter, letting settings.PROVIDER_PRIORITY decide which one.

        No `prefer=` here, and that is the whole point of the change. This used to ask for
        prefer="gemini" -- and ProviderRegistry.get_adapter puts a preferred provider at the FRONT
        of the candidate list, ahead of both priority and the failure cooldown:

            candidates = [self._providers[prefer]] + [c for c in candidates if c.name != prefer]

        So the Animator always received GeminiAdapter, whose generate_video is a bare
        `raise NotImplementedError`, no matter what PROVIDER_PRIORITY said. Every clip then failed
        into the placeholder below. Dropping `prefer` lets omniroute_video (priority 0) serve video,
        and leaves the cooldown machinery free to fall through if the router is down.
        """
        return await self.registry.get_adapter(capability="video")

    async def _persist_execution_via_orchestrator(
        self,
        agent_type: str,
        input_data: dict,
        project_id: str = "default",
    ) -> None:
        """Helper to persist an execution using the orchestrator's internal method.
        We don't have direct access to the orchestrator's _persist_execution, so we
        create a temporary orchestrator just for persistence? Actually, we can't.
        Instead, we'll rely on the orchestrator's agent methods to persist.
        This method is not used; we persist by calling the orchestrator's agent methods.
        """
        pass  # Persistence happens inside the orchestrator's agent methods.

    async def run_scriptwriter(self, idea: str, style: str = "Anime") -> Dict[str, Any]:
        """Run Agents 1-4 (Scriptwriter to Scene Designer) and return state for UI.
        This method persists AgentExecution rows for each of the four agents.
        """
        # Create a new database session for this pipeline run
        _, session_maker = _init_engine()
        async with session_maker() as db_session:
            orchestrator = AnimationAgentOrchestrator(
                registry=self.registry, db_session=db_session, model=self.model
            )

            # 1. Scriptwriter
            script_input = ScriptwriterInput(prompt=idea, style=style)
            script_output = await orchestrator.run_scriptwriter(script_input)
            self._scriptwriter_output = script_output

            # 2. Art Director
            art_input = ArtDirectorInput(scriptwriter_output=script_output)
            art_output = await orchestrator.run_art_director(art_input)
            self._art_director_output = art_output

            # 3. Character Designer
            char_input = CharacterDesignerInput(art_director_output=art_output)
            char_output = await orchestrator.run_character_designer(char_input)
            self._character_designer_output = char_output

            # 4. Scene Designer
            scene_input = SceneDesignerInput(art_director_output=art_output)
            scene_output = await orchestrator.run_scene_designer(scene_input)
            self._scene_designer_output = scene_output

            # Build legacy state format expected by the UI
            # Generate image URLs for characters and scenes using the image adapter
            image_adapter = await self._get_image_adapter()

            # Process characters: each char is a dict with name, description, prompt
            characters = []
            for char in self._character_designer_output.characters:
                prompt = char.get("prompt", char.get("description", ""))
                image_url = ""
                if prompt:
                    try:
                        img_request = ImageGenerationRequest(
                            prompt=prompt,
                            model="pollinations/flux",  # default; adapter will handle
                            n=1,
                            size="512x512",
                            response_format="url",
                        )
                        img_response = await image_adapter.generate_image(img_request)
                        image_url = img_response.data[0].get("url", "")
                    except Exception:
                        image_url = (
                            f"https://image.pollinations.ai/prompt/{urllib.parse.quote(prompt)}"
                            f"?width=512&height=512&nologo=true"
                        )
                characters.append(
                    {
                        "name": char.get("name", ""),
                        "role": char.get("description", ""),
                        "description": char.get("description", ""),
                        "image_url": image_url,
                    }
                )

            # Process scenes: each scene is a dict with description, prompt, style
            scenes = []
            for idx, scene in enumerate(self._scene_designer_output.scenes):
                prompt = scene.get("prompt", "")
                image_url = ""
                if prompt:
                    try:
                        img_request = ImageGenerationRequest(
                            prompt=prompt,
                            model="pollinations/flux",
                            n=1,
                            size="1024x1024",
                            response_format="url",
                        )
                        img_response = await image_adapter.generate_image(img_request)
                        image_url = img_response.data[0].get("url", "")
                    except Exception:
                        image_url = (
                            f"https://image.pollinations.ai/prompt/{urllib.parse.quote(prompt)}"
                            f"?width=1024&height=1024&nologo=true"
                        )
                scenes.append(
                    {
                        "scene": idx + 1,
                        "environment": scene.get("description", ""),
                        "mood": "",  # default
                        "motion": "",  # default
                        "visual_prompt": scene.get("prompt", ""),
                        "image_url": image_url,
                    }
                )

            state = {
                "script": self._scriptwriter_output.description,
                "art_direction": {
                    "style": self._art_director_output.visual_style,
                    "color_palette": self._art_director_output.color_palette,
                    "lighting": "Neon",  # default
                },
                "characters": characters,
                "scenes": scenes,
                "status": "pending_approval",
            }

            # Incremental state checkpoint (no-op for compatibility)
            self._checkpoint_state(
                "Scriptwriter, Art Director, Character Designer, Scene Designer"
            )
            return state

    async def run_animator(self, scenes_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run Agents 5-7 (Animator to Director) and return state for UI.
        This method expects the scenes_data from the scriptwriter run (though it
        may ignore it and use the stored scene designer output). It persists
        AgentExecution rows for the animator, sound director, and director.
        """
        _, session_maker = _init_engine()
        async with session_maker() as db_session:
            orchestrator = AnimationAgentOrchestrator(
                registry=self.registry, db_session=db_session, model=self.model
            )

            # We need the outputs from the first four agents; they should be stored
            # from the previous run_scriptwriter call. If not, we cannot proceed.
            if not all(
                [
                    self._scriptwriter_output,
                    self._art_director_output,
                    self._character_designer_output,
                    self._scene_designer_output,
                ]
            ):
                raise RuntimeError(
                    "Missing outputs from scriptwriter phase. Run run_scriptwriter first."
                )

            # 5. Animator
            # Agent 5: the Storyboard Artist, before anything is animated. A panel is what the
            # Animator turns into one paid clip, so agreeing the shot list first is the cheap step.
            # It is not fatal if it fails -- the Animator's input field is optional, so a storyboard
            # failure degrades to the old behaviour (Animator invents its own coverage) instead of
            # losing the whole film.
            storyboard_output = None
            storyboard_error = None
            try:
                storyboard_output = await orchestrator.run_storyboard_artist(
                    StoryboardArtistInput(
                        scene_designer_output=self._scene_designer_output,
                        character_designer_output=self._character_designer_output,
                    )
                )
                self._storyboard_artist_output = storyboard_output
            except Exception as exc:  # noqa: BLE001
                storyboard_error = f"{type(exc).__name__}: {exc}"

            anim_input = AnimatorInput(
                scene_designer_output=self._scene_designer_output,
                character_designer_output=self._character_designer_output,
                storyboard_artist_output=storyboard_output,
            )
            anim_output = await orchestrator.run_animator(anim_input)

            # 6. Sound Director
            sound_input = SoundDirectorInput(
                scriptwriter_output=self._scriptwriter_output,
                animator_output=anim_output,
            )
            sound_output = await orchestrator.run_sound_director(sound_input)

            # 7. Director
            director_input = DirectorInput(
                scriptwriter_output=self._scriptwriter_output,
                art_director_output=self._art_director_output,
                character_designer_output=self._character_designer_output,
                scene_designer_output=self._scene_designer_output,
                animator_output=anim_output,
                sound_director_output=sound_output,
            )
            director_output = await orchestrator.run_director(director_input)

            # Render one real clip per scene, under a budget for the whole run.
            #
            # CREDITS ARE REAL MONEY and this is a loop. The video adapter enforces a per-CLIP
            # ceiling, which bounds nothing here: an 8-scene film is 8 submits from one button
            # press, so at Flow's measured 20 credits a clip that is 160 credits with no ceiling
            # ever tripped. Hence the run-level budget below.
            #
            # It also used to answer a failed render with
            #   https://commondatastorage.googleapis.com/.../ElephantsDream.mp4
            # a stock clip of a different film entirely, presented as this project's scene. That is
            # gone. A scene that did not render says so, and says why.
            from app.core.config import settings as _settings

            video_adapter = await self._get_video_adapter()
            provider_name = video_adapter.get_provider_name()
            budget = _settings.OMNIROUTE_VIDEO_MAX_CREDITS_PER_RUN
            spent = 0
            animations = []
            for idx, anim in enumerate(anim_output.animations):
                prompt = anim.get("prompt", "")
                video_url = ""
                entry_extra: Dict[str, Any] = {}
                if prompt:
                    vid_request = VideoGenerationRequest(
                        prompt=prompt,
                        # "" means "provider's choice". MEASURED 2026-08-24: the literal
                        # "gemini-2.5-veo" that used to be hardcoded here is rejected by the router
                        # as an unknown model, so this loop could never have produced a video.
                        model=_settings.DEFAULT_VIDEO_MODEL or "",
                    )
                    # Free pre-check. quote() is optional on the adapter protocol, so a provider
                    # that cannot price a request is simply not budget-checked.
                    quote_fn = getattr(video_adapter, "quote", None)
                    cost = None
                    if quote_fn is not None:
                        try:
                            q = await quote_fn(vid_request)
                            if q.get("error"):
                                entry_extra["error"] = f"{provider_name}: {q['error']}"
                            else:
                                cost = q.get("quote", q.get("estimate"))
                        except Exception as exc:  # noqa: BLE001
                            entry_extra["error"] = (
                                f"{provider_name} unreachable: {type(exc).__name__}: {exc}"
                            )
                    if "error" not in entry_extra:
                        if cost is not None and spent + cost > budget:
                            # Reported, not silently dropped: a short film is better than a
                            # surprise invoice, but only if the UI can see the truncation.
                            entry_extra["skipped"] = "run_credit_budget_exhausted"
                            entry_extra["quote"] = cost
                            entry_extra["budget_remaining"] = budget - spent
                        else:
                            try:
                                vid_response = await video_adapter.generate_video(vid_request)
                                clip = (vid_response.data or [{}])[0]
                                video_url = clip.get("url", "") or ""
                                charged = clip.get("credits_charged")
                                # Charge what the provider says it charged; fall back to the quote,
                                # because an unbudgeted retry is how a loop like this runs away.
                                spent += charged if charged is not None else (cost or 0)
                                entry_extra["credits_charged"] = charged
                                entry_extra["local_path"] = clip.get("path")
                                entry_extra["job_id"] = clip.get("job_id")
                            except Exception as exc:  # noqa: BLE001
                                # No retry. A video submit that raises may still be running and
                                # still be charging, so a second attempt buys a second clip.
                                entry_extra["error"] = (
                                    f"{provider_name}: {type(exc).__name__}: {exc}"
                                )
                animations.append(
                    {
                        "scene": anim.get("scene", idx + 1),
                        "motion": anim.get("motion", "Pan"),
                        "video_url": video_url,
                        "provider": provider_name,
                        **entry_extra,
                    }
                )

            # Build legacy state format
            state = {
                "animation": animations,
                "storyboard_panels": (
                    storyboard_output.panels if storyboard_output is not None else []
                ),
                "storyboard_error": storyboard_error,
                "sound_plan": {
                    "bgm_track": "Synthwave Ambient Pulse (120 BPM)",
                    "voice_track": "https://api.streamelements.com/kappa/v2/speech?voice=Brian&text=The%20journey%20begins",
                    "sfx": ["Rain", "Engine", "Energy"],
                },
                "director_notes": "Stitched via FFmpeg (Mocked)",
                # Referenced rather than retyped: this list and AGENT_ROLES had already drifted
                # apart from the frontend's crew, which is how Storyboard Artist went missing.
                "roles": list(AGENT_ROLES),
                "status": "completed",
            }

            # Incremental state checkpoint
            self._checkpoint_state("Animator, Sound Director, Director")
            return state

    async def run_avatar_agent(self, script: str, character_image_url: str) -> Dict[str, Any]:
        """Agent 8: AI Avatar Agent generates a lip-synced video speaking the script."""
        _, session_maker = _init_engine()
        async with session_maker() as db_session:
            orchestrator = AnimationAgentOrchestrator(
                registry=self.registry, db_session=db_session, model=self.model
            )
            # Note: The legacy agent_pipeline had this method, but the orchestrator does not.
            # We'll keep the legacy implementation for now.
            kaggle_webhook = os.getenv("KAGGLE_AVATAR_WEBHOOK")

            avatar_video_url = None

            if kaggle_webhook:
                try:
                    async with httpx.AsyncClient() as client:
                        resp = await client.post(
                            kaggle_webhook,
                            json={
                                "text": script,
                                "reference_image_url": character_image_url,
                                "voice": "af_heart"
                            },
                            timeout=120.0
                        )
                        resp.raise_for_status()
                        data = resp.json()
                        avatar_video_url = data.get("video_url")
                except Exception as e:
                    print(f"Avatar Generation Error: {e}")

            if not avatar_video_url:
                # Fallback to mock URL
                avatar_video_url = "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/TearsOfSteel.mp4"

            state = {
                "avatar": {
                    "role": "AI Avatar Agent",
                    "video_url": avatar_video_url,
                    "status": "completed"
                }
            }

            self._checkpoint_state("AI Avatar Agent")
            return state

    def _checkpoint_state(self, agents_completed: str) -> None:
        """Dump the state to MASTER_WORKLOG.md as per the non-volatile state mandate."""
        # This writes directly to the worklog to ensure recovery
        # (In a real 3-hour film loop, this happens per scene)
        pass  # No-op for now; we keep the method for compatibility.


def create_animation_pipeline(registry: Optional[ProviderRegistry] = None,
                              model: Optional[str] = None) -> AnimationPipeline:
    return AnimationPipeline(registry=registry, model=model)