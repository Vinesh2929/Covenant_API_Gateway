"""
gateway/contracts/drift.py

Behavioral drift detection engine.

Tracks compliance scores over time per (app_id, contract_id) pair and
fires alerts when compliance degrades beyond a configurable threshold.

Architecture:
  - Redis sorted sets (ZSETs) store timestamped compliance scores.
    Key format: drift:{app_id}:{contract_id}
    Score = Unix timestamp (float), Member = "{timestamp}:{compliance_score}"
  - Rolling averages are computed over configurable windows (default 24h).
  - Drift is detected by comparing the recent window average against a
    longer baseline window (default 7 days).
  - When relative compliance drops > alert_threshold (default 10%), a
    DriftAlert is generated.

Why Redis sorted sets?
  ZADD is O(log N) per insert, ZRANGEBYSCORE is O(log N + M) for range
  queries.  This gives us efficient time-range queries without a dedicated
  time-series database.  The sorted set also makes cleanup trivial:
  ZREMRANGEBYSCORE to prune entries older than the retention window.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import structlog

log = structlog.get_logger()


@dataclass
class DriftAlert:
    """A detected compliance drift event."""
    app_id: str
    contract_id: str
    baseline_avg: float
    recent_avg: float
    drop_pct: float
    window_hours: float
    baseline_hours: float
    timestamp: float = field(default_factory=time.time)

    @property
    def severity(self) -> str:
        if self.drop_pct >= 0.25:
            return "critical"
        elif self.drop_pct >= 0.15:
            return "high"
        elif self.drop_pct >= 0.10:
            return "medium"
        return "low"


@dataclass
class ComplianceSnapshot:
    """Point-in-time compliance summary for a contract."""
    app_id: str
    contract_id: str
    avg_score: float
    sample_count: int
    min_score: float
    max_score: float
    window_hours: float


class DriftDetector:
    """
    Tracks compliance scores in Redis and detects behavioral drift.

    Usage::

        detector = DriftDetector(redis_client)

        # After every contract evaluation:
        await detector.record(app_id, contract_id, compliance_score)

        # Periodic check (e.g. every 5 minutes via background task):
        alerts = await detector.check_drift(app_id, contract_id)
        for alert in alerts:
            log.warning("Drift detected", **vars(alert))

        # Dashboard data:
        snapshot = await detector.get_snapshot(app_id, contract_id, hours=24)
    """

    def __init__(
        self,
        redis_client,
        key_prefix: str = "drift",
        recent_window_hours: float = 24.0,
        baseline_window_hours: float = 168.0,
        alert_threshold: float = 0.10,
        retention_days: int = 30,
        min_samples: int = 10,
    ) -> None:
        self._redis = redis_client
        self._prefix = key_prefix
        self._recent_hours = recent_window_hours
        self._baseline_hours = baseline_window_hours
        self._alert_threshold = alert_threshold
        self._retention_seconds = retention_days * 86400
        self._min_samples = min_samples

    def _key(self, app_id: str, contract_id: str) -> str:
        return f"{self._prefix}:{app_id}:{contract_id}"

    async def record(
        self,
        app_id: str,
        contract_id: str,
        compliance_score: float,
    ) -> None:
        """
        Record a compliance score for drift tracking.

        Stores the score in a Redis sorted set with the current timestamp
        as the sort key.  Also prunes entries older than the retention window.
        """
        now = time.time()
        key = self._key(app_id, contract_id)
        member = f"{now}:{compliance_score:.4f}"

        pipe = self._redis.pipeline()
        pipe.zadd(key, {member: now})

        # Prune entries older than retention window
        cutoff = now - self._retention_seconds
        pipe.zremrangebyscore(key, "-inf", cutoff)

        # Set a TTL on the key itself as a safety net
        pipe.expire(key, self._retention_seconds + 86400)

        await pipe.execute()

    async def get_scores_in_window(
        self,
        app_id: str,
        contract_id: str,
        hours: float,
    ) -> list[tuple[float, float]]:
        """
        Retrieve (timestamp, score) pairs from the last N hours.

        Returns:
            List of (unix_timestamp, compliance_score) tuples, ordered by time.
        """
        now = time.time()
        window_start = now - (hours * 3600)
        key = self._key(app_id, contract_id)

        members = await self._redis.zrangebyscore(key, window_start, now)

        results = []
        for member in members:
            try:
                ts_str, score_str = member.rsplit(":", 1)
                results.append((float(ts_str), float(score_str)))
            except (ValueError, IndexError):
                continue

        return results

    async def get_snapshot(
        self,
        app_id: str,
        contract_id: str,
        hours: float = 24.0,
    ) -> Optional[ComplianceSnapshot]:
        """Get a compliance summary for the specified time window."""
        scores = await self.get_scores_in_window(app_id, contract_id, hours)

        if not scores:
            return None

        values = [s for _, s in scores]
        return ComplianceSnapshot(
            app_id=app_id,
            contract_id=contract_id,
            avg_score=sum(values) / len(values),
            sample_count=len(values),
            min_score=min(values),
            max_score=max(values),
            window_hours=hours,
        )

    async def check_drift(
        self,
        app_id: str,
        contract_id: str,
    ) -> list[DriftAlert]:
        """
        Compare recent compliance against baseline and generate alerts.

        Algorithm:
          1. Compute average compliance over the recent window (default 24h).
          2. Compute average compliance over the baseline window (default 7d).
          3. If (baseline - recent) / baseline > alert_threshold, fire alert.

        Only fires if both windows have >= min_samples data points to avoid
        false positives on sparse data.
        """
        recent_scores = await self.get_scores_in_window(
            app_id, contract_id, self._recent_hours
        )
        baseline_scores = await self.get_scores_in_window(
            app_id, contract_id, self._baseline_hours
        )

        if len(recent_scores) < self._min_samples:
            return []
        if len(baseline_scores) < self._min_samples:
            return []

        recent_values = [s for _, s in recent_scores]
        baseline_values = [s for _, s in baseline_scores]

        recent_avg = sum(recent_values) / len(recent_values)
        baseline_avg = sum(baseline_values) / len(baseline_values)

        if baseline_avg <= 0:
            return []

        # Relative drop: how much has compliance decreased compared to baseline
        drop = (baseline_avg - recent_avg) / baseline_avg

        if drop > self._alert_threshold:
            alert = DriftAlert(
                app_id=app_id,
                contract_id=contract_id,
                baseline_avg=round(baseline_avg, 4),
                recent_avg=round(recent_avg, 4),
                drop_pct=round(drop, 4),
                window_hours=self._recent_hours,
                baseline_hours=self._baseline_hours,
            )

            log.warning(
                "Behavioral drift detected",
                app_id=app_id,
                contract_id=contract_id,
                baseline_avg=alert.baseline_avg,
                recent_avg=alert.recent_avg,
                drop_pct=f"{alert.drop_pct:.1%}",
                severity=alert.severity,
            )

            return [alert]

        return []

    async def check_all_drift(
        self,
        app_id: str,
        contract_ids: list[str],
    ) -> list[DriftAlert]:
        """Check drift for all contracts of an application."""
        import asyncio
        results = await asyncio.gather(
            *[self.check_drift(app_id, cid) for cid in contract_ids],
            return_exceptions=True,
        )

        alerts = []
        for result in results:
            if isinstance(result, list):
                alerts.extend(result)
            elif isinstance(result, Exception):
                log.error("Drift check failed", error=str(result))

        return alerts

    async def get_drift_summary(
        self,
        app_id: str,
        contract_ids: list[str],
    ) -> dict:
        """
        Build a summary for the drift dashboard endpoint.

        Returns a dict with per-contract compliance snapshots and any
        active drift alerts.
        """
        import asyncio

        snapshot_tasks = [
            self.get_snapshot(app_id, cid, hours=self._recent_hours)
            for cid in contract_ids
        ]
        alert_tasks = [
            self.check_drift(app_id, cid)
            for cid in contract_ids
        ]

        snapshots = await asyncio.gather(*snapshot_tasks, return_exceptions=True)
        alerts = await asyncio.gather(*alert_tasks, return_exceptions=True)

        contracts_summary = {}
        all_alerts = []

        for cid, snap, alert_result in zip(contract_ids, snapshots, alerts):
            if isinstance(snap, ComplianceSnapshot):
                contracts_summary[cid] = {
                    "avg_compliance": round(snap.avg_score, 4),
                    "sample_count": snap.sample_count,
                    "min_score": round(snap.min_score, 4),
                    "max_score": round(snap.max_score, 4),
                    "window_hours": snap.window_hours,
                }
            else:
                contracts_summary[cid] = {"avg_compliance": None, "sample_count": 0}

            if isinstance(alert_result, list):
                for a in alert_result:
                    all_alerts.append({
                        "contract_id": a.contract_id,
                        "severity": a.severity,
                        "baseline_avg": a.baseline_avg,
                        "recent_avg": a.recent_avg,
                        "drop_pct": f"{a.drop_pct:.1%}",
                    })

        return {
            "app_id": app_id,
            "window_hours": self._recent_hours,
            "baseline_hours": self._baseline_hours,
            "contracts": contracts_summary,
            "alerts": all_alerts,
            "alert_count": len(all_alerts),
        }
