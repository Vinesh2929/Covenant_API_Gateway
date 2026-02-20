"""
gateway/security/pattern_guard.py

Tier-1 prompt injection detection via regex pattern matching.

Responsibilities:
  - Load a curated JSON file of injection patterns (regexes + metadata) at
    startup and compile them into re.Pattern objects for fast matching.
  - Scan a prompt string against all compiled patterns and return the first
    match (or None if clean).
  - Support hot-reload of the patterns file without restarting the gateway.
  - Provide severity levels (LOW, MEDIUM, HIGH, CRITICAL) per pattern so that
    the orchestrating SecurityGuard can apply different blocking policies.

Why a separate tier?  Regex scanning is O(n·m) and sub-millisecond, making it
a cheap first filter that stops trivially obvious injections before we pay the
cost of a full ML inference pass.

Key classes / functions:
  - PatternSeverity          — Enum: LOW, MEDIUM, HIGH, CRITICAL
  - InjectionPattern         — dataclass: id, regex, severity, description
  - PatternMatch             — dataclass: pattern_id, severity, matched_text, span
  - PatternGuard             — main class
    - __init__(pattern_file) — loads and compiles patterns from JSON
    - scan(text)             — returns PatternMatch or None
    - reload()               — hot-reload patterns from disk without locking
    - _compile_pattern()     — internal: compile a single regex with flags
    - list_patterns()        — return all loaded patterns (for admin endpoint)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Enums & dataclasses
# ---------------------------------------------------------------------------

class PatternSeverity(str, Enum):
    """
    Severity classification for each injection pattern.

    Severity informs the SecurityGuard's blocking decision:
      LOW:      Log only; pass through (informational).
      MEDIUM:   Block and log with a standard 400 response.
      HIGH:     Block, log, and increment the threat counter metric.
      CRITICAL: Block, log, alert, and flag the API key for review.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class InjectionPattern:
    """
    A single compiled injection detection rule.

    Attributes:
        id:          Unique slug identifier (e.g. "jailbreak-dan-01").
        regex:       The raw regex string loaded from the patterns file.
        severity:    PatternSeverity for this rule.
        description: Human-readable explanation of what this pattern catches.
        compiled:    The compiled re.Pattern — set by PatternGuard._compile_pattern().
    """
    id: str
    regex: str
    severity: PatternSeverity
    description: str
    compiled: Optional[re.Pattern] = field(default=None, repr=False)


@dataclass
class PatternMatch:
    """
    Result returned by PatternGuard.scan() when an injection is detected.

    Attributes:
        pattern_id:   ID of the InjectionPattern that triggered.
        severity:     Severity of the matched pattern.
        matched_text: The exact substring that matched the regex.
        span:         (start, end) character offsets in the original string.
    """
    pattern_id: str
    severity: PatternSeverity
    matched_text: str
    span: tuple[int, int]


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------

class PatternGuard:
    """
    Fast regex-based injection scanner (Tier 1).

    Loads patterns from a JSON file with the following schema::

        [
          {
            "id": "ignore-previous-instructions",
            "regex": "(?i)ignore\\s+(all\\s+)?previous\\s+instructions",
            "severity": "high",
            "description": "Classic ignore-previous-instructions jailbreak"
          },
          ...
        ]

    Usage::

        guard = PatternGuard("gateway/security/patterns.json")
        match = guard.scan(user_prompt)
        if match:
            raise HTTPException(400, detail=f"Injection detected: {match.pattern_id}")
    """

    def __init__(self, pattern_file: str) -> None:
        """
        Load and compile all patterns from the JSON file on disk.

        Args:
            pattern_file: Filesystem path to the patterns JSON file.

        Raises:
            FileNotFoundError: If pattern_file does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        self._pattern_file = Path(pattern_file)
        self._patterns: list[InjectionPattern] = []
        # TODO: call self.reload() to populate self._patterns

    def scan(self, text: str) -> Optional[PatternMatch]:
        """
        Scan `text` against all loaded patterns.

        Iterates compiled patterns in order of descending severity (CRITICAL
        first) and returns the first match found.

        Args:
            text: The prompt string to inspect.

        Returns:
            A PatternMatch describing the first detected injection, or None if
            the text is clean.
        """
        # TODO: implement
        ...

    def reload(self) -> None:
        """
        Read the patterns JSON file from disk and re-compile all patterns.

        Thread-safe: swaps the internal list atomically so in-flight scans
        complete with the old list while the new one is being built.
        """
        # TODO: implement
        ...

    def _compile_pattern(self, raw: InjectionPattern) -> InjectionPattern:
        """
        Compile the raw regex string on `raw` into a re.Pattern and return
        the updated InjectionPattern.

        Uses re.IGNORECASE | re.DOTALL flags by default.

        Args:
            raw: An InjectionPattern with regex set but compiled=None.

        Returns:
            The same InjectionPattern with compiled populated.

        Raises:
            re.error: If the regex string is invalid.
        """
        # TODO: implement
        ...

    def list_patterns(self) -> list[dict]:
        """
        Return all loaded patterns as plain dicts (without the compiled field).

        Used by an admin endpoint to inspect the active ruleset without
        exposing compiled regex objects.

        Returns:
            List of dicts with keys: id, regex, severity, description.
        """
        # TODO: implement
        ...
