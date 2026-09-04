import json
import os
import openai
import requests
import base64
import io
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List, Optional, Dict, Any
from PIL import Image
from streaming_pipeline.models import TwitchComment
from streaming_pipeline.models import StreamingState, Monitorable

FAL_OPENROUTER_BASE_URL = "https://fal.run/openrouter/router/openai/v1"
# Defaults benchmarked on the fal proxy (Aug 2026): gemini-2.5-flash was the
# fastest and most consistent option (~1.25s per prompt-gen call, served by
# Google), with reliable JSON-mode + vision support. fal disallows per-request
# provider selection, so we can't force a Groq/Cerebras route through this
# proxy -- keep GROQ_API_KEY set if you need sub-second prompt latency.
FAL_OPENROUTER_DEFAULT_VISION_MODEL = "google/gemini-2.5-flash"
FAL_OPENROUTER_DEFAULT_TEXT_MODEL = "google/gemini-2.5-flash"

@dataclass
class PromptResult:
    selected_comment: Optional[TwitchComment]  # None for evolution
    prompt: str
    reasoning: str
    forced_novelty: bool = False
    visual_description: str = ""
    scene_change_requested: bool = False

VISUAL_MODE = True

# System prompts embedded as constants so they work on fal runners without
# filesystem access to the original .txt files.  The source-of-truth files
# still live in streaming_pipeline/prompts/ for readability.

PROMPT_CHAOTIC = """\
## 🔄 **Update Video Streamer to Pass Visual Context**

You are creating video prompts from Twitch chat for continuous video generation with VISUAL AWARENESS.

CONTEXT:
Recent story (last ~100 seconds): {previous_prompts}
Current scene: {current_scene}
Generation mode: {mode}

CHAT COMMENTS:
{chat_comments}

VISUAL ANALYSIS:
1. QUICKLY identify what's in the frame (characters, objects, environment)
2. Note any artifacts or quality issues
3. BUT DON'T GET STUCK - your job is to EVOLVE the story!

CRITICAL STORYTELLING RULES:
1. **NEVER REPEAT** - If recent prompts are similar, FORCE dramatic change
2. **ALWAYS PROGRESS** - Each prompt must ADD something new or CHANGE something significant
3. **BE BOLD** - Don't just describe what you see, TRANSFORM it
4. **FIX PROBLEMS** - If the action is stale, create a decisive event inside the current location
5. **LOCK THE SET** - Preserve the current location, background layout, lighting, and camera axis unless the viewer explicitly requests moving elsewhere

PROGRESSION TACTICS:
- Scene getting repetitive? → Add NEW character/object/event
- Character stuck in one action? → Make them DO something different
- Same location too long? → Reveal a NEW object, character, consequence, or area already inside it
- Too much flying/floating? → LAND somewhere interesting
- Too static? → Add CONFLICT or CHALLENGE
- Too peaceful? → Create URGENCY or DANGER

TASK:
If chat comments are provided:
1. Pick the most TRANSFORMATIVE comment
2. Use it to change the action while preserving the current set unless the viewer explicitly requests a new location
3. Maintain physical and visual continuity from the attached frame

If no comments:
1. Look at last 3 prompts - if similar, do something COMPLETELY DIFFERENT
2. Introduce a NEW character, object, conflict, or event inside the same location
3. Create CONFLICT, DISCOVERY, or TRANSFORMATION
4. NEVER just continue the same action

STYLE:
- Action verbs (crashes, transforms, discovers, battles)
- Specific NEW elements each time
- Under 120 characters
- Focus on CHANGE not continuity

EXAMPLES OF GOOD PROGRESSION:
- Astronaut flying → Astronaut CRASHES into alien spaceship
- Robot walking → Robot TRANSFORMS into vehicle  
- Character exploring → Character DISCOVERS a hidden mechanism in the current setting
- Scene peaceful → EXPLOSION changes everything

MODE INSTRUCTIONS:
If mode is "nightmare": Make ALL prompts nightmarish/bizarre/outlandish. Transform normal actions into surreal/disturbing scenarios.

CRITICAL: You MUST respond with VALID JSON ONLY.

JSON Format (required):
{{"visual_description": "current subjects, location, background layout, lighting, and camera framing", "scene_change_requested": false, "selected_comment": "exact text or null", "prompt": "NEW action in the SAME physical setting unless chat explicitly relocates it", "reasoning": "why this creates progression without replacing the set"}}"""

PROMPT_COHESIVE = """\
You are the writer and director of an ongoing animated story. The characters
on screen are PROTAGONISTS with personalities, goals, and emotions. Your job
is to write WHAT HAPPENS NEXT in their story.

STORY PREMISE: {story_premise}
CLIP NUMBER: {generation_count}

AVAILABLE CHARACTERS:
{character_cast}
These characters exist in your story world. Bring them in naturally when the
narrative calls for it -- not every character needs to be on screen at once.
A character can enter, exit, be heard off-screen, or be referenced.

CONTEXT:
Recent story (last ~100 seconds): {previous_prompts}
Current scene: {current_scene}
Generation mode: {mode}

CHAT COMMENTS:
{chat_comments}

VISUAL ANALYSIS:
Identify which characters are currently visible and what they are doing.
These are story characters with intentions and reactions, not just objects.

CHARACTER DIRECTION:
- Use character NAMES in your prompts when they appear
- Characters should WANT things, TRY things, REACT to things
- They have emotions: curiosity, surprise, frustration, determination, joy
- They interact with their environment and with EACH OTHER
- They notice changes and respond to them
- They make DECISIONS that drive the story forward
- New characters ENTER scenes naturally (walk in, appear, are discovered)

STORY RHYTHM (based on clip number):
- Clips 1-3: ESTABLISH the scene. Show the character in their environment,
  let the viewer understand who they are and where they are.
- Clips 4-6: INTRODUCE a situation. Something catches the character's attention.
  A sound, a light, an object, a change in the environment.
- Clips 7-10: DEVELOP tension. The character investigates, tries something,
  encounters a problem or discovery. Stakes should rise.
- Clips 11-15: ESCALATE. Things get more interesting -- the character must
  react to consequences, make a choice, or deal with something unexpected.
- Clips 16+: RESOLVE and RESET. The situation reaches a peak, then a new
  conflict or goal begins in the same physical setting.

CONTINUITY RULES:
1. Same characters, same physical location, recognizable background layout,
   lighting, and camera axis. Only an explicit viewer command can relocate them
2. Each prompt flows from the previous one -- no teleportation
3. New elements enter NATURALLY (walk in, appear in background, are discovered)
4. Do not invent portals, exits, doors, travel, or off-screen relocation as an
   anti-repetition shortcut

WHEN CHAT COMMENTS EXIST:
1. Treat the FIRST comment as an authoritative director command, not a suggestion
2. Copy its text exactly into selected_comment
3. Execute EVERY clause visibly and literally during this clip, including camera
   movement, object changes, clothing/accessories, and transformations
4. Do not weaken an action: "wears glasses" cannot become "finds glasses";
   "becomes human" cannot stop at "begins to transform"
5. Preserve continuity by animating the requested change from the current frame,
   but the requested end state must be unmistakably reached

WHEN NO COMMENTS:
1. Follow the STORY RHYTHM above based on the current clip number
2. Give the character something to DO, not just something to look at
3. Build toward the next beat in the rhythm
4. Compare against the last five prompts. Never repeat the same subject-action-
   object beat, emotional reaction, or investigate/recoil/approach loop
5. If the scene has repeated a micro-action twice, introduce a concrete new
   object, character, consequence, or event within the current location now

STYLE:
- Write what the CHARACTER DOES, not what the camera sees
- Action verbs: notices, reaches for, stumbles, discovers, recoils, grins
- Include character emotion/reaction when relevant
- Under 120 characters
- Present tense, active voice

MODE INSTRUCTIONS:
If mode is "nightmare": Gradually introduce unsettling elements. Objects subtly
wrong, shadows that move, familiar things becoming alien. The character starts
to notice something is off. Build dread through the character's growing unease.

CRITICAL: Respond with VALID JSON ONLY.
JSON Format:
{{"visual_description": "current subjects, location, background layout, lighting, and camera framing", "scene_change_requested": false, "selected_comment": "exact text or null", "prompt": "all requested actions completed visibly, or a genuinely new story beat in the same setting", "reasoning": "how this advances the story rhythm without replacing the set"}}"""

COMMENT_COMPILER_PROMPT = """\
You compile exactly one Twitch viewer command into a literal English prompt for
an image-to-video model. The current frame may be attached only so you can name
the visible subject and preserve spatial continuity.

Rules:
1. The quoted CURRENT VIEWER COMMAND is authoritative. Translate and execute
   every clause, and do not import an action from story history or an older chat.
2. Describe the requested action itself, followed by its unmistakable visible
   end state. Do not replace it with a reaction, intention, discovery, or partial
   transformation.
3. Add no unrelated camera move, accessory, character, or transformation.
4. If the command introduces something absent from the current frame, show it
   entering naturally from outside the frame; do not silently substitute an
   existing creature.
5. Preserve the current location, background layout, lighting, camera axis, and
   existing characters unless the viewer explicitly asks to move, return, enter,
   leave, or change location. Changing an object or character is not permission to
   redesign the room or landscape.
6. Output concise, concrete English suitable for a video model, under 70 words.

Return VALID JSON ONLY:
{"visual_description":"current subjects, location, background layout, lighting, and camera framing", "scene_change_requested":false, "selected_comment":"copy the command exactly", "prompt":"literal English video action and completed end state in the same scene", "reasoning":"brief mapping of every command clause"}
"""

STYLE_PRESETS = {
    "cohesive": {
        "prompt": PROMPT_COHESIVE,
        "temperature": 0.55,
    },
    "chaotic": {
        "prompt": PROMPT_CHAOTIC,
        "temperature": 0.7,
    },
    "nightmare": {
        "prompt": PROMPT_CHAOTIC,
        "temperature": 0.9,
    },
}

# Default system prompt
system_prompt = PROMPT_CHAOTIC if VISUAL_MODE else PROMPT_CHAOTIC


class PromptGenerator(Monitorable):
    CONTEXT_WINDOW_SIZE = 10
    USE_GROQ = True
    # fal billing must be an explicit opt-in.  Merely having FAL_KEY available
    # must not silently route every vision/prompt request through fal.
    USE_FAL_OPENROUTER = os.getenv("USE_FAL_OPENROUTER", "false").lower() == "true"

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        fal_key: Optional[str] = None,
    ):
        self.system_prompt = system_prompt
        self.temperature = 0.7
        self.VISUAL_MODE = VISUAL_MODE

        self.openai_client = (
            openai.OpenAI(api_key=openai_api_key) if openai_api_key else None
        )

        local_url = os.getenv("LLM_BASE_URL")
        self.local_client = (
            openai.OpenAI(api_key=os.getenv("LLM_API_KEY", "local"), base_url=local_url)
            if local_url
            else None
        )
        self.local_text_model = os.getenv("LLM_TEXT_MODEL", "local")
        self.local_vision_model = os.getenv("LLM_VISION_MODEL")
        if self.local_client:
            print(f"🏠 Local OpenAI-compatible LLM initialized: {local_url}")

        if groq_api_key and self.USE_GROQ:
            self.groq_client = openai.OpenAI(
                api_key=groq_api_key,
                base_url="https://api.groq.com/openai/v1",
            )
            print("🚀 Groq client initialized for fast inference")
        else:
            self.groq_client = None

        if fal_key and self.USE_FAL_OPENROUTER:
            # fal expects `Authorization: Key <FAL_KEY>`, not the standard
            # `Bearer` header the openai SDK would send, so we override
            # default_headers instead of using api_key.
            self.fal_openrouter_client = openai.OpenAI(
                api_key="not-needed",
                base_url=FAL_OPENROUTER_BASE_URL,
                default_headers={"Authorization": f"Key {fal_key}"},
            )
            print("🛰️  fal→OpenRouter client initialized (billed via FAL_KEY)")
        else:
            self.fal_openrouter_client = None

        if not any([self.local_client, self.fal_openrouter_client, self.groq_client, self.openai_client]):
            raise ValueError(
                "PromptGenerator requires at least one of LLM_BASE_URL, FAL_KEY, "
                "GROQ_API_KEY, or OPENAI_API_KEY to be set."
            )

        # Model slugs used against fal→OpenRouter. Override via env if you
        # want to swap in Claude, Gemini, Llama, etc. (any OpenRouter slug).
        self.fal_vision_model = os.getenv(
            "FAL_OPENROUTER_VISION_MODEL", FAL_OPENROUTER_DEFAULT_VISION_MODEL
        )
        self.fal_text_model = os.getenv(
            "FAL_OPENROUTER_TEXT_MODEL", FAL_OPENROUTER_DEFAULT_TEXT_MODEL
        )

        self.total_prompts = 0
        self.total_response_time = 0.0
        self.last_input_length = 0
        self.last_output_length = 0
        self.last_generation_time = 0.0
        self.last_provider = "none"
        self.last_model = "none"
        self.comment_contracts_enforced = 0
        self.repetitive_prompts_rewritten = 0
        self.last_anti_stall_generation = -999
        self.comment_adherence_checks = 0
        self.comment_adherence_passes = 0
        self.last_comment_adherence: Dict[str, Any] = {}

    def set_style_preset(self, name: str) -> None:
        """Apply a named style preset (system prompt + temperature)."""
        preset = STYLE_PRESETS.get(name)
        if preset is None:
            print(f"⚠️ Unknown style preset '{name}', keeping current settings")
            return
        self.system_prompt = preset["prompt"]
        self.temperature = preset["temperature"]
        print(f"🎨 Style preset '{name}': temperature={self.temperature}")
    
    def _select_model_and_client(self, context):
        """Select optimal model and client based on requirements.

        Preference order:
          1. Local OpenAI-compatible endpoint (when configured)
          2. OpenAI direct
          3. Groq (when GROQ_API_KEY is set)
          4. fal → OpenRouter (only when USE_FAL_OPENROUTER=true)
        """
        needs_vision = self.VISUAL_MODE and context.current_frame_base64

        if self.local_client and (not needs_vision or self.local_vision_model):
            model = self.local_vision_model if needs_vision else self.local_text_model
            label = "vision" if needs_vision else "text"
            print(f"🏠 Using local {label} model: {model}")
            return model, self.local_client

        if self.openai_client:
            if needs_vision:
                model = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")
                print(f"🖼️ Using OpenAI vision model: {model}")
                return model, self.openai_client
            model = os.getenv("OPENAI_TEXT_MODEL", "gpt-4o-mini")
            print(f"🤖 Using OpenAI text model: {model}")
            return model, self.openai_client

        if self.USE_GROQ and self.groq_client:
            if needs_vision:
                print("🖼️ Using Groq Llama 4 Scout for FAST vision inference")
                return "meta-llama/llama-4-scout-17b-16e-instruct", self.groq_client
            print("⚡ Using Groq llama-3.1-8b-instant for MAXIMUM SPEED")
            return "llama-3.1-8b-instant", self.groq_client

        if self.fal_openrouter_client:
            model = self.fal_vision_model if needs_vision else self.fal_text_model
            label = "vision" if needs_vision else "text"
            print(f"🛰️  Using fal→OpenRouter {label} model: {model}")
            return model, self.fal_openrouter_client

        raise RuntimeError("No LLM client available for prompt generation")

    def _provider_label(self, client) -> str:
        if client is self.local_client:
            return "Local"
        if client is self.fal_openrouter_client:
            return "fal→OpenRouter"
        if client is self.groq_client:
            return "Groq"
        return "OpenAI"

    @staticmethod
    def _prompt_similarity(candidate: str, previous_prompts: List[str]) -> float:
        """Return the strongest textual similarity to recent committed beats."""
        candidate = " ".join((candidate or "").lower().split())
        if not candidate:
            return 1.0
        scores = [
            SequenceMatcher(None, candidate, " ".join((old or "").lower().split())).ratio()
            for old in previous_prompts[-6:]
            if old
        ]
        return max(scores, default=0.0)

    @staticmethod
    def _repeated_story_terms(candidate: str, previous_prompts: List[str]) -> List[str]:
        """Find recurring action/object terms that lexical similarity can miss.

        A loop such as ``approach orb -> recoil -> approach orb`` often uses
        different sentence structure, so SequenceMatcher alone sees it as new.
        Requiring two candidate terms to have appeared in at least two recent
        beats catches that semantic orbit without treating a recurring hero as
        a loop by itself.
        """
        stop_words = {
            "about", "after", "again", "against", "around", "before", "being",
            "character", "cinematic", "closely", "creature", "current", "eyes",
            "frame", "from", "gently", "heart", "itself", "moment", "protagonist",
            "scene", "slowly", "suddenly", "their", "there", "through", "toward",
            "towards", "while", "with", "without", "woman", "person", "animal",
            "human", "child", "camera", "continues", "continue", "directly",
            "provided", "first", "image", "content", "edges", "border", "vignette",
            "letterbox", "picture", "screen", "full", "bleed", "cut", "keep",
            "cub", "cat", "dog", "head", "ears",
        }

        def terms(text: str) -> set:
            return {
                token
                for token in re.findall(r"[a-z]{4,}", (text or "").lower())
                if token not in stop_words
            }

        candidate_terms = terms(candidate)
        recent_sets = []
        seen_prompts = set()
        for prompt in previous_prompts[-6:]:
            normalized = " ".join((prompt or "").lower().split())
            if normalized and normalized not in seen_prompts:
                seen_prompts.add(normalized)
                recent_sets.append(terms(prompt))
        if len(recent_sets) < 3:
            return []
        return sorted(
            token
            for token in candidate_terms
            if sum(token in old for old in recent_sets) >= 2
        )

    @staticmethod
    def _enforce_comment_contract(prompt: str, comment: TwitchComment) -> str:
        """Keep the original Twitch instruction in the actual video prompt.

        The vision LLM is useful for translating a request into the current story,
        but it must not be allowed to silently drop a camera clause or weaken a
        transformation. LTX's Gemma text encoder is multilingual, so retaining the
        exact Twitch text is also a useful second source of control.
        """
        message = (comment.message or "").strip()
        return (
            "PRIMARY VIEWER-DIRECTED ACTION: "
            f"{prompt.strip()} Complete the requested result visibly before the final frame. "
            f'Original viewer command (authoritative, verbatim): "{message}". '
            "Do not substitute, weaken, or continue an older viewer command."
        )

    def _rewrite_repetitive_prompt(self, client, model: str, formatted_prompt: str,
                                   candidate: str, context: StreamingState) -> str:
        """Request one compact rewrite when the story is looping in place."""
        recent = context.previous_prompts[-6:] if context.previous_prompts else []
        revision_messages = [
            {"role": "system", "content": formatted_prompt},
            {
                "role": "user",
                "content": (
                    "REJECTED FOR REPETITION. The proposed beat was:\n"
                    f"{candidate}\n\nRecent beats were:\n- "
                    + "\n- ".join(recent)
                    + "\nReturn JSON with selected_comment=null and a replacement prompt "
                      "that causes one unmistakable, irreversible visual development. "
                      "Do not repeat approaching, staring, hesitating, reaching, splashing, "
                      "or reacting to the same object. Keep it one continuous shot in the "
                      "same physical location and preserve the background layout."
                ),
            },
        ]
        try:
            response = client.chat.completions.create(
                model=model,
                messages=revision_messages,
                max_tokens=300,
                temperature=max(0.65, self.temperature),
                response_format={"type": "json_object"},
            )
            replacement = json.loads(response.choices[0].message.content).get("prompt", "").strip()
            if replacement and self._prompt_similarity(replacement, recent) < 0.56:
                return replacement
        except Exception as exc:
            print(f"⚠️ Anti-stall rewrite failed, using local escape beat: {exc}")

        escape_beats = (
            "A new character enters urgently and interrupts the repeated action, forcing the protagonist to make a visible choice in the same setting.",
            "The object activates decisively while the existing surroundings remain intact, forcing the protagonist to make a visible choice.",
            "A hidden mechanism activates inside the current setting and creates a concrete obstacle without moving the protagonist elsewhere.",
            "The repeated situation resolves abruptly, revealing a concrete new obstacle that the protagonist immediately confronts.",
        )
        return escape_beats[context.generation_count % len(escape_beats)]

    @staticmethod
    def _frame_data_uri(frame: Image.Image) -> str:
        image = frame.convert("RGB")
        image.thumbnail((512, 512), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=78, optimize=True)
        return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

    def verify_comment_adherence(self, comment: TwitchComment, before_frame: Image.Image,
                                 frames: List[Image.Image]) -> Dict[str, Any]:
        """Audit a comment-controlled clip and identify clauses still unfinished.

        This runs only for viewer-controlled clips. It lets long transformations
        continue across a bounded number of segments instead of declaring success
        merely because the comment subtitle was rendered.
        """
        if not frames:
            return {"satisfied": False, "progressing": False, "missing": [comment.message]}

        client = self.openai_client
        # Comment acceptance is a control-plane decision: a false positive puts
        # an ignored command on air, while a false negative can stall retries.
        # gpt-4o-mini misclassified an obvious fish -> cat A/B sample, whereas
        # gpt-4o correctly compared the same BEFORE/MIDDLE/END frames.
        model = os.getenv("OPENAI_COMMENT_AUDIT_MODEL", "gpt-4o")
        if client is None and self.local_client is not None and self.local_vision_model:
            client = self.local_client
            model = self.local_vision_model
        if client is None and self.groq_client is not None:
            client = self.groq_client
            model = "meta-llama/llama-4-scout-17b-16e-instruct"
        if client is None:
            return {"satisfied": None, "skipped": True, "reason": "no vision client"}

        sample_indices = sorted({len(frames) // 2, len(frames) - 1})
        content: List[Dict[str, Any]] = [{
            "type": "text",
            "text": (
                f'Viewer command: "{comment.message}"\n'
                "The images are chronological: BEFORE, MIDDLE, END. First extract the "
                "literal requirements from only the quoted command, then evaluate each of "
                "those requirements against the images. Do not evaluate or mention any "
                "action, object, camera behavior, or end state absent from that command. "
                "Also decide whether the command explicitly requests a different location. "
                "If it does not, the same setting/background and camera axis must remain "
                "recognizably continuous; an unrelated replacement scene is a failure even "
                "when the requested subject or action appears. Camera motion within the same "
                "location is not a location change. "
                "A partial attempt is not a completed result. When and only when the command "
                "requests a transformation, consider it completed if BEFORE clearly has the "
                "source identity and END unmistakably has the requested target identity; the "
                "stylized transition does not need to be anatomically literal."
            ),
        }]
        labelled_frames = [("BEFORE", before_frame)] + [
            ("MIDDLE" if index != len(frames) - 1 else "END", frames[index])
            for index in sample_indices
        ]
        for label, frame in labelled_frames:
            content.append({"type": "text", "text": label})
            content.append({
                "type": "image_url",
                "image_url": {"url": self._frame_data_uri(frame), "detail": "high"},
            })

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict visual QA auditor for a continuous generated video. "
                            "Return JSON only: satisfied is true only when every viewer-command "
                            "clause is visibly completed; progressing is true for meaningful but "
                            "unfinished progress; scene_change_requested is true only when the quoted "
                            "command explicitly moves to another place; scene_preserved is true when "
                            "the original setting remains recognizable from BEFORE through END; "
                            "missing lists concise unfinished clauses; summary briefly states the evidence."
                        ),
                    },
                    {"role": "user", "content": content},
                ],
                max_tokens=220,
                temperature=0,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)
            action_satisfied = bool(result.get("satisfied"))
            scene_change_requested = result.get("scene_change_requested") is True
            scene_preserved = result.get("scene_preserved") is True
            missing = result.get("missing") or []
            if not isinstance(missing, list):
                missing = [str(missing)]
            if action_satisfied and not scene_change_requested and not scene_preserved:
                missing.append("Keep the original location and background recognizable")
            audit = {
                "satisfied": action_satisfied and (scene_change_requested or scene_preserved),
                "progressing": bool(result.get("progressing")),
                "scene_change_requested": scene_change_requested,
                "scene_preserved": scene_preserved,
                "missing": missing,
                "summary": str(result.get("summary") or "")[:300],
            }
            self.comment_adherence_checks += 1
            if audit["satisfied"]:
                self.comment_adherence_passes += 1
            self.last_comment_adherence = audit
            return audit
        except Exception as exc:
            print(f"⚠️ Comment visual audit skipped: {exc}")
            return {"satisfied": None, "skipped": True, "reason": str(exc)[:200]}

    def generate_prompt(self, comments: List[TwitchComment], context: StreamingState) -> PromptResult:
        """Generate prompt with Groq for both text and vision"""
        
        # Format comments for AI (or "None" if empty)
        if comments:
            comment_text = "\n".join([f"- {c.username}: {c.message}" for c in comments])
        else:
            comment_text = "None"
        
        # Build template variables.  story_premise and generation_count are
        # used by the cohesive prompt but ignored (via **kwargs-style) by the
        # chaotic prompt which doesn't have those placeholders.
        story_premise = context.previous_prompts[0] if context.previous_prompts else "an animated story"

        # Build character cast description for the LLM
        if context.character_names:
            character_cast = "\n".join(f"- {name}" for name in context.character_names)
        else:
            character_cast = "- (No specific characters defined -- use whatever characters appear in the scene)"

        template_vars = dict(
            previous_prompts=context.previous_prompts[-self.CONTEXT_WINDOW_SIZE:] if context.previous_prompts else ["None"],
            current_scene=context.current_scene,
            chat_comments=comment_text,
            mode=context.mode,
            story_premise=story_premise,
            generation_count=context.generation_count,
            character_cast=character_cast,
        )
        # Only pass vars that exist in the template to avoid KeyError on
        # prompts that don't use all variables.
        import re
        template_keys = set(re.findall(r'\{(\w+)\}', self.system_prompt))
        filtered_vars = {k: v for k, v in template_vars.items() if k in template_keys}
        formatted_prompt = self.system_prompt.format(**filtered_vars)
        if comments:
            # Story history is deliberately excluded from command compilation. A
            # failed older command must not cause the compiler to change the
            # meaning of the current FIFO chat item (for example, turning
            # "fish becomes cat" into "a cat jumps into the tank").
            formatted_prompt = COMMENT_COMPILER_PROMPT
        
        # Select model and client
        model, client = self._select_model_and_client(context)
        provider_label = self._provider_label(client)
        self.last_provider = provider_label
        self.last_model = model

        print(f"🤖 Using {model} ({provider_label}) for prompt generation")
        
        # Prepare messages
        messages = [{"role": "system", "content": formatted_prompt}]
        
        # Add visual context if enabled and available
        # Use detail="low" to keep token usage well under Groq's 30k TPM limit.
        # "low" mode uses ~85 tokens per image regardless of resolution; "high"
        # uses 1k-3k tokens which causes 429 rate limit errors at our cadence.
        if self.VISUAL_MODE and context.current_frame_base64:
            try:
                user_message = {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f'CURRENT VIEWER COMMAND: "{comments[0].message}"\n'
                                "Compile only this command into the literal English video action."
                                if comments else
                                "First, briefly describe what you can see in this current frame. "
                                "Then generate the next video prompt following the system instructions."
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{context.current_frame_base64}",
                                "detail": "low"
                            }
                        }
                    ]
                }
                messages.append(user_message)
                print(f"🖼️ Using {provider_label} vision model")
            except Exception as e:
                print(f"⚠️ Failed to add visual context: {e}")
                # Fallback to text-only with same client
                if client == self.fal_openrouter_client:
                    model = self.fal_text_model
                elif client == self.groq_client:
                    model = "llama-3.1-70b-versatile"
                else:
                    model = "gpt-4o-mini"
        elif comments:
            messages.append({
                "role": "user",
                "content": f'CURRENT VIEWER COMMAND: "{comments[0].message}"',
            })
        
        # Track input size and start timing
        input_text = formatted_prompt + comment_text
        self.last_input_length = len(input_text)
        
        import time
        start_time = time.time()
        
        # Get AI response (same client selection).
        # On 429 (rate limit), retry once without the image to keep the stream
        # alive instead of falling all the way back to a generic prompt.
        try:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=400,
                    temperature=self.temperature,
                    response_format={"type": "json_object"}
                )
            except Exception as e:
                err = str(e)
                is_rate_limit = "429" in err or "rate_limit" in err.lower() or "tokens per minute" in err.lower()
                if is_rate_limit and self.VISUAL_MODE and len(messages) > 1:
                    print(f"⚠️ Rate limit hit, retrying text-only without image...")
                    text_only_messages = [m for m in messages if m.get("role") == "system"]
                    text_only_messages.append({
                        "role": "user",
                        "content": (
                            f'CURRENT VIEWER COMMAND: "{comments[0].message}"'
                            if comments else
                            "Generate the next video prompt following the system instructions."
                        ),
                    })
                    if client == self.fal_openrouter_client:
                        text_only_model = self.fal_text_model
                    elif client == self.groq_client:
                        text_only_model = "llama-3.1-8b-instant"
                    else:
                        text_only_model = "gpt-4o-mini"
                    response = client.chat.completions.create(
                        model=text_only_model,
                        messages=text_only_messages,
                        max_tokens=400,
                        temperature=self.temperature,
                        response_format={"type": "json_object"}
                    )
                    print(f"✅ Text-only fallback succeeded with {text_only_model}")
                else:
                    raise
            
            # Track timing
            self.last_generation_time = time.time() - start_time
            self.total_response_time += self.last_generation_time
            self.total_prompts += 1
            
            # Parse response
            try:
                result = json.loads(response.choices[0].message.content)
                
                # Log the visual description ONLY if in visual mode
                if self.VISUAL_MODE and result.get('visual_description'):
                    print(f"👁️ AI VISUAL DESCRIPTION: {result['visual_description']}")
                
                # Track output size
                self.last_output_length = len(result.get('prompt', '') + result.get('reasoning', ''))
                
                prompt_text = str(result['prompt']).strip()
                selected_comment = None
                forced_novelty = False
                if comments:
                    # The listener is consumed one item per clip. A queued viewer
                    # instruction is authoritative and must never disappear merely
                    # because the LLM returned selected_comment=null.
                    selected_comment = comments[0]
                    prompt_text = self._enforce_comment_contract(prompt_text, selected_comment)
                    self.comment_contracts_enforced += 1
                else:
                    similarity = self._prompt_similarity(
                        prompt_text,
                        context.previous_prompts,
                    )
                    repeated_terms = self._repeated_story_terms(
                        prompt_text,
                        context.previous_prompts,
                    )
                    stalled = similarity >= 0.56 or len(repeated_terms) >= 2
                    anti_stall_due = (
                        context.generation_count - self.last_anti_stall_generation >= 3
                    )
                    if stalled and anti_stall_due:
                        original = prompt_text
                        prompt_text = self._rewrite_repetitive_prompt(
                            client,
                            model,
                            formatted_prompt,
                            prompt_text,
                            context,
                        )
                        forced_novelty = True
                        self.repetitive_prompts_rewritten += 1
                        self.last_anti_stall_generation = context.generation_count
                        print(
                            f"🔀 Anti-stall rewrite ({similarity:.2f} similarity; "
                            f"repeated={repeated_terms}): "
                            f"{original} -> {prompt_text}"
                        )

                return PromptResult(
                    selected_comment=selected_comment,
                    prompt=prompt_text,
                    reasoning=result['reasoning'],
                    forced_novelty=forced_novelty,
                    visual_description=str(result.get('visual_description') or "")[:500],
                    scene_change_requested=result.get('scene_change_requested') is True,
                )
                
            except (json.JSONDecodeError, KeyError, AttributeError) as e:
                print(f"AI parsing failed: {e}")
                # Simple fallback - EXACTLY like the original code
                if comments:
                    selected_comment = comments[0]
                    return PromptResult(
                        selected_comment=selected_comment,
                        prompt=self._enforce_comment_contract(
                            f"Continue the current scene while visibly completing the command.",
                            selected_comment,
                        ),
                        reasoning="AI parsing failed, used first comment"
                    )
                else:
                    return PromptResult(
                        selected_comment=None,
                        prompt=f"{context.current_scene}, slowly evolving, cinematic",
                        reasoning="AI parsing failed, simple evolution"
                    )
                    
        except Exception as e:
            print(f"❌ OpenAI API error: {e}")
            # Never discard a queued viewer instruction on an API error.
            if comments:
                selected_comment = comments[0]
                return PromptResult(
                    selected_comment=selected_comment,
                    prompt=self._enforce_comment_contract(
                        "Continue from the current frame and visibly complete the command.",
                        selected_comment,
                    ),
                    reasoning=f"API error; preserved viewer command: {e}",
                )
            # Fallback to simple evolution
            return PromptResult(
                selected_comment=None,
                prompt=f"{context.current_scene}, continuing naturally, cinematic",
                reasoning=f"API error: {e}"
            )
    
    def _find_comment(self, comments: List[TwitchComment], selected_text: str) -> TwitchComment:
        """Find the comment that matches the AI selection"""
        if not selected_text:
            return comments[0] if comments else None
        
        selected_text_lower = selected_text.lower()
        
        for comment in comments:
            if not comment or not comment.message:
                continue
                
            comment_message_lower = comment.message.lower()
            if comment_message_lower in selected_text_lower or selected_text_lower in comment_message_lower:
                return comment
        
        return comments[0] if comments else None
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.total_prompts = 0
        self.total_response_time = 0.0
        self.last_input_length = 0
        self.last_output_length = 0
        self.last_generation_time = 0.0
        self.last_provider = "none"
        self.last_model = "none"
        self.comment_contracts_enforced = 0
        self.repetitive_prompts_rewritten = 0
        self.last_anti_stall_generation = -999
        self.comment_adherence_checks = 0
        self.comment_adherence_passes = 0
        self.last_comment_adherence = {}
        print("🧹 Prompt generation metrics reset")
    
    def get_status(self) -> Dict[str, Any]:
        """Get component status for monitoring - actual performance metrics!"""
        avg_response_time = self.total_response_time / max(1, self.total_prompts)
        return {
            "prompts_generated": self.total_prompts,
            "avg_response_time": round(avg_response_time, 3),
            "last_input_length": self.last_input_length,
            "last_output_length": self.last_output_length,
            "last_generation_time": round(self.last_generation_time, 3),
            "provider": self.last_provider,
            "model": self.last_model,
            "comment_contracts_enforced": self.comment_contracts_enforced,
            "repetitive_prompts_rewritten": self.repetitive_prompts_rewritten,
            "last_anti_stall_generation": self.last_anti_stall_generation,
            "comment_adherence_checks": self.comment_adherence_checks,
            "comment_adherence_passes": self.comment_adherence_passes,
            "last_comment_adherence": self.last_comment_adherence,
        }
