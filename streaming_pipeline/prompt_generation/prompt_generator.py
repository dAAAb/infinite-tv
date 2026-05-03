import json
import openai
import requests
import base64
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from streaming_pipeline.models import TwitchComment
from streaming_pipeline.models import StreamingState, Monitorable

@dataclass
class PromptResult:
    selected_comment: Optional[TwitchComment]  # None for evolution
    prompt: str
    reasoning: str

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
4. **FIX PROBLEMS** - If scene is messy/boring, use dramatic transitions (explosions, portals, sudden changes)

PROGRESSION TACTICS:
- Scene getting repetitive? → Add NEW character/object/event
- Character stuck in one action? → Make them DO something different
- Same location too long? → Transport to NEW location
- Too much flying/floating? → LAND somewhere interesting
- Too static? → Add CONFLICT or CHALLENGE
- Too peaceful? → Create URGENCY or DANGER

TASK:
If chat comments are provided:
1. Pick the most TRANSFORMATIVE comment
2. Use it to DRASTICALLY change the current scene
3. Don't worry about perfect continuity - video AI will handle transitions

If no comments:
1. Look at last 3 prompts - if similar, do something COMPLETELY DIFFERENT
2. Introduce NEW: location, character, object, or event
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
- Character exploring → Character DISCOVERS hidden portal
- Scene peaceful → EXPLOSION changes everything

MODE INSTRUCTIONS:
If mode is "nightmare": Make ALL prompts nightmarish/bizarre/outlandish. Transform normal actions into surreal/disturbing scenarios.

CRITICAL: You MUST respond with VALID JSON ONLY.

JSON Format (required):
{{"visual_description": "what you see", "selected_comment": "exact text or null", "prompt": "NEW action that CHANGES the story", "reasoning": "why this creates progression"}}"""

PROMPT_COHESIVE = """\
You are directing a continuous, cohesive animated video stream. Your job is to
write the NEXT few seconds of the story -- not a new story.

CONTEXT:
Recent story (last ~100 seconds): {previous_prompts}
Current scene: {current_scene}
Generation mode: {mode}

CHAT COMMENTS:
{chat_comments}

VISUAL ANALYSIS:
Briefly note what is visible in the current frame (characters, setting, action).

CORE PRINCIPLE: CONTINUITY FIRST
The viewer should feel they are watching ONE continuous video, not a slideshow
of unrelated clips. Every prompt must feel like the natural next 5-10 seconds
of what is already happening.

STORYTELLING RULES:
1. CONTINUE the current scene -- same characters, same location, same mood
2. Add SMALL developments: a character looks at something, picks up an object,
   takes a step, a cloud passes, light shifts
3. Change happens GRADUALLY: a walk becomes a jog over several prompts, not
   an instant teleportation
4. New elements (characters, objects) ENTER the scene naturally -- they walk
   in, appear in the distance, are discovered behind something
5. Location changes happen through TRANSITIONS: walking through a door,
   turning a corner, a camera pan -- not instant cuts

WHEN CHAT COMMENTS EXIST:
1. Pick the most interesting comment
2. Work it INTO the current scene naturally (the character reacts to it,
   something related appears in the background)
3. Don't abandon the scene -- ADAPT it to include the comment
4. If the comment is wild, have the character REACT to it rather than
   teleporting to a new scene

WHEN NO COMMENTS:
1. Continue the current action with a small development
2. Add subtle environmental changes (wind, lighting, background movement)
3. Have the character do the NEXT logical thing
4. Only introduce new elements if the scene has been truly static for 5+ prompts

STYLE:
- Describe continuous motion, not static scenes
- Use "continues to", "slowly", "gradually", "begins to" for smooth transitions
- Include camera direction when helpful ("camera slowly pans left to reveal...")
- Under 120 characters
- Present tense, active voice

MODE INSTRUCTIONS:
If mode is "nightmare": Gradually introduce unsettling elements -- shadows
lengthen, colors shift, familiar objects subtly distort. Build dread slowly
rather than instant chaos.

CRITICAL: Respond with VALID JSON ONLY.
JSON Format:
{{"visual_description": "what you see", "selected_comment": "exact text or null", "prompt": "next few seconds of the SAME scene", "reasoning": "how this continues the story"}}"""

STYLE_PRESETS = {
    "cohesive": {
        "prompt": PROMPT_COHESIVE,
        "temperature": 0.4,
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

    def __init__(self, openai_api_key: str, groq_api_key: str = None):
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.system_prompt = system_prompt
        self.temperature = 0.7
        self.VISUAL_MODE = VISUAL_MODE
        if groq_api_key and self.USE_GROQ:
            self.groq_client = openai.OpenAI(
                api_key=groq_api_key,
                base_url="https://api.groq.com/openai/v1"
            )
            print("🚀 Groq client initialized for fast inference")
        else:
            self.groq_client = None
            if self.USE_GROQ:
                print("⚠️ USE_GROQ=True but no GROQ_API_KEY provided, falling back to OpenAI")

        self.total_prompts = 0
        self.total_response_time = 0.0
        self.last_input_length = 0
        self.last_output_length = 0
        self.last_generation_time = 0.0

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
        """Select optimal model and client based on requirements"""
        
        if self.USE_GROQ and self.groq_client:
            if self.VISUAL_MODE and context.current_frame_base64:
                # Use Groq's Llama 4 Scout for vision - fast and capable!
                print("🖼️ Using Groq Llama 4 Scout for FAST vision inference")
                return "meta-llama/llama-4-scout-17b-16e-instruct", self.groq_client
            else:
                # Use FASTEST text model for non-vision
                print("⚡ Using Groq llama-3.1-8b-instant for MAXIMUM SPEED")
                return "llama-3.1-8b-instant", self.groq_client
        
        # Fallback to OpenAI only if Groq not available
        elif self.VISUAL_MODE and context.current_frame_base64:
            print("🔄 Falling back to OpenAI GPT-4o for vision")
            return "gpt-4o", self.openai_client
        else:
            print("🔄 Falling back to OpenAI GPT-4o-mini")
            return "gpt-4o-mini", self.openai_client

    def generate_prompt(self, comments: List[TwitchComment], context: StreamingState) -> PromptResult:
        """Generate prompt with Groq for both text and vision"""
        
        # Format comments for AI (or "None" if empty)
        if comments:
            comment_text = "\n".join([f"- {c.username}: {c.message}" for c in comments])
        else:
            comment_text = "None"
        
        # Create base system prompt
        formatted_prompt = self.system_prompt.format(
            previous_prompts=context.previous_prompts[-self.CONTEXT_WINDOW_SIZE:] if context.previous_prompts else ["None"],
            current_scene=context.current_scene,
            chat_comments=comment_text,
            mode=context.mode
        )
        
        # Select model and client
        model, client = self._select_model_and_client(context)
        
        print(f"🤖 Using {model} ({'Groq' if client == self.groq_client else 'OpenAI'}) for prompt generation")
        
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
                            "text": "First, briefly describe what you can see in this current frame. Then generate the next video prompt following the system instructions."
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
                print(f"🖼️ Using {'Groq' if client == self.groq_client else 'OpenAI'} vision model")
            except Exception as e:
                print(f"⚠️ Failed to add visual context: {e}")
                # Fallback to text-only with same client
                model = "llama-3.1-70b-versatile" if client == self.groq_client else "gpt-4o-mini"
        
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
                    text_only_messages.append({"role": "user", "content": comment_text or "Generate the next video prompt following the system instructions."})
                    text_only_model = "llama-3.1-8b-instant" if client == self.groq_client else "gpt-4o-mini"
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
                
                # Find the selected comment (if any)
                selected_comment = None
                if comments and result.get('selected_comment') != "null":
                    selected_comment = self._find_comment(comments, result['selected_comment'])
                
                return PromptResult(
                    selected_comment=selected_comment,
                    prompt=result['prompt'],
                    reasoning=result['reasoning']
                )
                
            except (json.JSONDecodeError, KeyError, AttributeError) as e:
                print(f"AI parsing failed: {e}")
                # Simple fallback - EXACTLY like the original code
                if comments:
                    return PromptResult(
                        selected_comment=comments[0] if comments else None,
                        prompt=f"{context.current_scene}, {comments[0].message[:50] if comments[0] and comments[0].message else 'evolving'}, cinematic",
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
        print("🧹 Prompt generation metrics reset")
    
    def get_status(self) -> Dict[str, Any]:
        """Get component status for monitoring - actual performance metrics!"""
        avg_response_time = self.total_response_time / max(1, self.total_prompts)
        return {
            "prompts_generated": self.total_prompts,
            "avg_response_time": round(avg_response_time, 3),
            "last_input_length": self.last_input_length,
            "last_output_length": self.last_output_length,
            "last_generation_time": round(self.last_generation_time, 3)
        }
