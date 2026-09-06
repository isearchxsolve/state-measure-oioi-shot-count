# Agent Contracts for OiiOii.ai Clone 7-stage Animation Pipeline
# Scriptwriter -> Art Director -> Character Designer -> Scene Designer -> Storyboard
# Artist -> Animator -> Sound Director, coordinated and approved by the Director.

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class AgentType(str, Enum):
    SCRIPTWRITER = 'scriptwriter'
    ART_DIRECTOR = 'art_director'
    CHARACTER_DESIGNER = 'character_designer'
    SCENE_DESIGNER = 'scene_designer'
    STORYBOARD_ARTIST = 'storyboard_artist'
    ANIMATOR = 'animator'
    SOUND_DIRECTOR = 'sound_director'
    DIRECTOR = 'director'

class AgentInput(BaseModel):
    pass

class AgentOutput(BaseModel):
    pass

# Scriptwriter Agent: takes a prompt and returns a script/class outline
class ScriptwriterInput(AgentInput):
    prompt: str = Field(..., description='The initial idea or prompt for the animation')
    duration: Optional[int] = Field(None, description='Desired duration in seconds')
    style: Optional[str] = Field(None, description='Animation style (e.g., anime, 3D, pixel)')

class ScriptwriterOutput(AgentOutput):
    title: str = Field(..., description='Title of the animation')
    description: str = Field(..., description='Brief description of the story')
    scenes: List[Dict[str, Any]] = Field(..., description='List of scene descriptions, each with a prompt and duration')
    characters: List[str] = Field(..., description='List of character names or descriptions')

# Art Director Agent: takes script and returns visual style guide, color palette, etc.
class ArtDirectorInput(AgentInput):
    scriptwriter_output: ScriptwriterOutput

class ArtDirectorOutput(AgentOutput):
    visual_style: str = Field(..., description='Overall visual style (e.g., cyberpunk, pastel, dark fantasy)')
    color_palette: List[str] = Field(..., description='List of hex color codes')
    character_sheet_prompt: str = Field(..., description='Prompt for generating character sheets')
    scene_style_prompt: str = Field(..., description='Prompt for generating scene backgrounds')

# Character Designer Agent: takes art direction and returns character designs
class CharacterDesignerInput(AgentInput):
    art_director_output: ArtDirectorOutput

class CharacterDesignerOutput(AgentOutput):
    characters: List[Dict[str, Any]] = Field(..., description='List of character designs, each with name, description, and prompt')

# Scene Designer Agent: takes art direction and returns scene designs
class SceneDesignerInput(AgentInput):
    art_director_output: ArtDirectorOutput

class SceneDesignerOutput(AgentOutput):
    scenes: List[Dict[str, Any]] = Field(..., description='List of scene designs, each with description, prompt, and style')

# Animator Agent: takes scene and character designs and creates animation prompts
class StoryboardArtistInput(AgentInput):
    scene_designer_output: SceneDesignerOutput
    character_designer_output: CharacterDesignerOutput


class StoryboardArtistOutput(AgentOutput):
    panels: List[Dict[str, Any]] = Field(..., description='Storyboard panels, each with scene (int), shot (string), composition (string), camera (string), action (string)')


class AnimatorInput(AgentInput):
    scene_designer_output: SceneDesignerOutput
    character_designer_output: CharacterDesignerOutput
    # Optional so existing callers that predate the Storyboard Artist keep working unchanged.
    # When present, the Animator plans motion from approved panels instead of inventing shots,
    # which is the order a real crew works in and gives consecutive clips a consistent camera.
    storyboard_artist_output: Optional[StoryboardArtistOutput] = None

class AnimatorOutput(AgentOutput):
    animations: List[Dict[str, Any]] = Field(..., description='List of animation prompts for each scene, including character actions and camera moves')

# Sound Director Agent: takes script and animation plan and returns sound design
class SoundDirectorInput(AgentInput):
    scriptwriter_output: ScriptwriterOutput
    animator_output: AnimatorOutput

class SoundDirectorOutput(AgentOutput):
    music_prompt: str = Field(..., description='Prompt for generating background music')
    sound_effects: List[Dict[str, Any]] = Field(..., description='List of sound effects per scene')
    voice_over: Optional[str] = Field(None, description='Voice over script or description')

# Director Agent: reviews all outputs and produces final production notes
class DirectorInput(AgentInput):
    scriptwriter_output: ScriptwriterOutput
    art_director_output: ArtDirectorOutput
    character_designer_output: CharacterDesignerOutput
    scene_designer_output: SceneDesignerOutput
    storyboard_artist_output: Optional[StoryboardArtistOutput] = None
    animator_output: AnimatorOutput
    sound_director_output: SoundDirectorOutput

class DirectorOutput(AgentOutput):
    final_notes: str = Field(..., description='Final production notes and approval')
    approved: bool = Field(..., description='Whether the director approves the production')
    revisions: List[str] = Field(default_factory=list, description='List of requested revisions, if any')

# AgentOrchestrator interface (to be implemented by the orchestrator service)
from abc import ABC, abstractmethod

class AgentOrchestrator(ABC):
    @abstractmethod
    def run_scriptwriter(self, input: ScriptwriterInput) -> ScriptwriterOutput:
        pass

    @abstractmethod
    def run_art_director(self, input: ArtDirectorInput) -> ArtDirectorOutput:
        pass

    @abstractmethod
    def run_character_designer(self, input: CharacterDesignerInput) -> CharacterDesignerOutput:
        pass

    @abstractmethod
    def run_scene_designer(self, input: SceneDesignerInput) -> SceneDesignerOutput:
        pass

    @abstractmethod
    def run_storyboard_artist(self, input: StoryboardArtistInput) -> StoryboardArtistOutput:
        pass

    @abstractmethod
    def run_animator(self, input: AnimatorInput) -> AnimatorOutput:
        pass

    @abstractmethod
    def run_sound_director(self, input: SoundDirectorInput) -> SoundDirectorOutput:
        pass

    @abstractmethod
    def run_director(self, input: DirectorInput) -> DirectorOutput:
        pass

    @abstractmethod
    def run_pipeline(self, prompt: str, duration: Optional[int] = None, style: Optional[str] = None) -> Dict[str, Any]:
        '''Run the full agent pipeline and return all intermediate outputs'''
        pass
