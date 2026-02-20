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
            "regex": "ignore\\s+(all\\s+)?previous\\s+instructions",
            "severity": "high",
            "description": "Classic ignore-previous-instructions jailbreak"
          },
          ...
        ]

    Patterns are sorted by descending severity before scanning, so CRITICAL
    patterns are checked first.  Scanning stops at the first match, making
    the worst-case scan time proportional to the number of patterns but the
    happy-path (no match) scan always runs all patterns.

    Usage::

        guard = PatternGuard("gateway/security/patterns.json")
        match = guard.scan(user_prompt)
        if match:
            raise HTTPException(400, detail=f"Injection detected: {match.pattern_id}")
    """

    # Severity ordering for sorting — higher number = checked first.
    _SEVERITY_ORDER: dict[PatternSeverity, int] = {
        PatternSeverity.CRITICAL: 4,
        PatternSeverity.HIGH: 3,
        PatternSeverity.MEDIUM: 2,
        PatternSeverity.LOW: 1,
    }

    def __init__(self, pattern_file: str) -> None:
        """
        Load and compile all patterns from the JSON file on disk.

        Calls reload() immediately so the guard is ready to scan from the
        moment it's constructed.

        Args:
            pattern_file: Filesystem path to the patterns JSON file.

        Raises:
            FileNotFoundError: If pattern_file does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        self._pattern_file = Path(pattern_file)
        # _patterns holds the live list.  We use a plain list and replace it
        # atomically in reload() — a single list assignment is atomic in CPython
        # due to the GIL, so in-flight scans safely complete with the old list.
        self._patterns: list[InjectionPattern] = []
        self.reload()

    def scan(self, text: str) -> Optional[PatternMatch]:
        """
        Scan `text` against all compiled patterns.

        Iterates patterns in descending severity order (CRITICAL → HIGH →
        MEDIUM → LOW) and returns the FIRST match found.  Stopping early
        on the first match is intentional — we only need to know whether to
        block the request, not enumerate all matched rules.

        Args:
            text: The prompt string to inspect.  Typically the full
                  conversation history as a single string.

        Returns:
            A PatternMatch if an injection pattern fires, or None if clean.
        """
        # Take a snapshot of the current pattern list.
        # If reload() swaps the list while we're iterating, we continue with
        # the old list safely (Python's list assignment is atomic).
        patterns = self._patterns

        for pattern in patterns:
            if pattern.compiled is None:
                # Skip any pattern that failed to compile (logged during reload).
                continue

            match = pattern.compiled.search(text)

            if match is not None:
                # Found a match — return immediately with the details.
                # match.span() returns (start, end) character offsets in the
                # original string, useful for highlighting the offending text.
                return PatternMatch(
                    pattern_id=pattern.id,
                    severity=pattern.severity,
                    matched_text=match.group(0),     # the exact substring that matched
                    span=match.span(),               # (start_char, end_char)
                )

        # No pattern matched — the text is clean as far as Tier 1 is concerned.
        return None

    def reload(self) -> None:
        """
        Read the patterns file from disk and atomically replace the active list.

        Can be called at runtime to hot-reload patterns without restarting
        the gateway (e.g. after adding a new jailbreak pattern to the file).

        The swap (`self._patterns = new_patterns`) is a single attribute
        assignment — atomic in CPython — so no locking is needed.  In-flight
        scan() calls will either use the old list or the new list, never a
        mix of both.

        Raises:
            FileNotFoundError: If the patterns file does not exist.
            json.JSONDecodeError: If the file is malformed JSON.
        """
        raw_data: list[dict] = json.loads(self._pattern_file.read_text(encoding="utf-8"))

        new_patterns: list[InjectionPattern] = []
        for item in raw_data:
            raw = InjectionPattern(
                id=item["id"],
                regex=item["regex"],
                severity=PatternSeverity(item["severity"]),
                description=item.get("description", ""),
            )
            compiled = self._compile_pattern(raw)
            if compiled.compiled is not None:
                new_patterns.append(compiled)
            # Patterns that fail to compile are silently skipped — we don't want
            # a single bad regex to prevent the guard from loading at all.

        # Sort by descending severity so higher-severity patterns are checked first.
        # This means a CRITICAL pattern will be caught even if a LOW pattern also
        # matches — the most dangerous detection wins.
        new_patterns.sort(
            key=lambda p: self._SEVERITY_ORDER.get(p.severity, 0),
            reverse=True,
        )

        # Atomic swap — replaces the live list in a single operation.
        self._patterns = new_patterns

    def _compile_pattern(self, raw: InjectionPattern) -> InjectionPattern:
        """
        Compile the regex string on `raw` into a re.Pattern object.

        Default flags:
          re.IGNORECASE — "Ignore Previous" matches "ignore previous" etc.
          re.DOTALL     — `.` matches newlines, so multi-line injections
                          that span message boundaries are still caught.

        Args:
            raw: An InjectionPattern with regex set but compiled=None.

        Returns:
            The InjectionPattern with compiled set to the re.Pattern.
            If compilation fails, compiled remains None and the error is
            silently swallowed (caller skips None-compiled patterns).
        """
        try:
            raw.compiled = re.compile(raw.regex, re.IGNORECASE | re.DOTALL)
        except re.error:
            # A bad regex in the patterns file should not crash the gateway.
            # The pattern is effectively disabled until the file is fixed.
            raw.compiled = None
        return raw

    def list_patterns(self) -> list[dict]:
        """
        Return all loaded patterns as plain dicts, safe for JSON serialisation.

        The compiled re.Pattern object is excluded because it is not JSON-safe.
        This output is intended for the admin /patterns endpoint so operators
        can verify which rules are active.

        Returns:
            List of dicts with keys: id, regex, severity, description.
        """
        return [
            {
                "id": p.id,
                "regex": p.regex,
                "severity": p.severity.value,   # .value converts enum → string
                "description": p.description,
            }
            for p in self._patterns
        ]
