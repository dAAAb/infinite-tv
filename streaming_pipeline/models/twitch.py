from dataclasses import dataclass
from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class UserCommentParams(BaseModel):
    """Parameter overrides when using user comments"""
    # LTX 2.5 distilled runs at CFG 1.0; the Comfy graph intentionally ignores
    # higher CFG values. Prompt adherence is controlled here by weakening the
    # first-frame guide while frame zero is still replaced with the exact handoff.
    guidance_scale: float = Field(default=1.0, description="LTX 2.5 distilled CFG")
    strength: float = Field(default=0.30, description="First-attempt image guide for visible viewer-requested changes")


@dataclass
class TwitchComment:
    username: str
    message: str
    timestamp: float
    user_id: Optional[str] = None
    badges: Optional[List[str]] = None
    emotes: Optional[Dict] = None
