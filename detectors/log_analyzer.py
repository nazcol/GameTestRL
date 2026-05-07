#log / telemetry pattern recognition.

# sub analyzers
# 1. consoleLogAnalyzer —>> TF-IDF + k-means clustering of JS console messages
# 2. performanceAnalyzer —>> frame timing anomaly detection
# 3. bugPrioritizer  —>> ranks detected issues by severity


import re
import logging
import time
from collections import Counter
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


#helpers

_ERROR_PATTERNS = [
    (re.compile(r"TypeError", re.I), "type_error", 9),
    (re.compile(r"ReferenceError", re.I), "reference_error", 8),
    (re.compile(r"Cannot read prop", re.I), "null_deref", 9),
    (re.compile(r"is not a function", re.I), "not_a_function", 8),
    (re.compile(r"undefined", re.I), "undefined_access", 6),
    (re.compile(r"NaN", re.I), "nan_value", 7),
    (re.compile(r"Uncaught", re.I), "uncaught_exception", 10),
    (re.compile(r"memory|heap|stack overflow", re.I), "memory_issue", 10),
    (re.compile(r"infinite loop|maximum call stack", re.I), "infinite_loop", 10),
    (re.compile(r"animation|render|frame", re.I), "render_issue", 5),
    (re.compile(r"score|tile|board|merge", re.I), "game_logic", 7),
]


def _classify_message(text: str) -> tuple[str, int]:
    for pattern, category, severity in _ERROR_PATTERNS:
        if pattern.search(text):
            return category, severity
    return "info", 1


# console / JS log analyzer                                           
class ConsoleLogAnalyzer:
    """
    Clusters console messages with TF-IDF + k-means.
    Identifies recurring error patterns and rare one-off anomalies.
    """

    def __init__(self, n_clusters: int = 8):
        self.n_clusters = n_clusters
        self._vectorizer = TfidfVectorizer(
            max_features=200,
            ngram_range=(1, 2),
            stop_words="english",
        )
        self._kmeans: Optional[KMeans] = None
        self._messages: list[dict] = []

    def ingest(self, telemetry: dict):
        """Feed raw telemetry dict from GameDriver.get_telemetry()."""
        for entry in telemetry.get("console_logs", []):
            entry["category"], entry["severity"] = _classify_message(entry["text"])
            self._messages.append(entry)
        for entry in telemetry.get("js_errors", []):
            entry["category"] = "js_error"
            entry["severity"] = 10
            self._messages.append(entry)

    def analyze(self) -> dict:
        if not self._messages:
            return {"clusters": [], "top_errors": [], "total_messages": 0}

        texts = [m["text"] for m in self._messages]
        cats = [m.get("category", "info") for m in self._messages]
        severities = [m.get("severity", 1) for m in self._messages]

        # cluster if we have enough messages
        cluster_labels = []
        if len(texts) >= self.n_clusters:
            try:
                X = self._vectorizer.fit_transform(texts)
                k = min(self.n_clusters, len(texts))
                self._kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
                cluster_labels = self._kmeans.fit_predict(X).tolist()
            except Exception as e:
                logger.warning("Clustering failed: %s", e)

        # category breakdown
        cat_counts = Counter(cats)
        sev_max = max(severities) if severities else 0

        # top errors by severity
        error_msgs = [
            m for m in self._messages
            if m.get("severity", 0) >= 7
        ]
        error_msgs.sort(key=lambda m: m.get("severity", 0), reverse=True)

        # cluster summary
        cluster_summary = []
        if cluster_labels:
            for cluster_id in range(max(cluster_labels) + 1):
                idxs = [i for i, l in enumerate(cluster_labels) if l == cluster_id]
                cluster_texts = [texts[i] for i in idxs]
                cluster_sev = [severities[i] for i in idxs]
                cluster_summary.append({
                    "id": cluster_id,
                    "count": len(idxs),
                    "max_severity": max(cluster_sev),
                    "sample": cluster_texts[0][:120],
                })
            cluster_summary.sort(key=lambda c: c["max_severity"], reverse=True)

        return {
            "total_messages": len(self._messages),
            "category_counts": dict(cat_counts),
            "max_severity": sev_max,
            "clusters": cluster_summary,
            "top_errors": [
                {"text": m["text"][:200], "category": m["category"], "severity": m["severity"]}
                for m in error_msgs[:20]
            ],
        }


# Performance / timing analyzer                                        
class PerformanceAnalyzer:

    def __init__(self, hitch_multiplier: float = 2.5):
        self.hitch_multiplier = hitch_multiplier
        self._timestamps: list[float] = []
        self._hitches: list[dict] = []

    def record_frame(self):
        now = time.time()
        self._timestamps.append(now)

        if len(self._timestamps) > 2:
            delta = now - self._timestamps[-2]
            if len(self._timestamps) > 10:
                recent = np.diff(self._timestamps[-20:])
                avg = recent.mean()
                if delta > avg * self.hitch_multiplier:
                    self._hitches.append({
                        "ts": now,
                        "delta_ms": delta * 1000,
                        "avg_ms": avg * 1000,
                        "ratio": delta / avg,
                    })

    def summary(self) -> dict:
        if len(self._timestamps) < 2:
            return {"frames": len(self._timestamps), "hitches": 0, "avg_frame_ms": 0}

        deltas = np.diff(self._timestamps) * 1000
        return {
            "frames": len(self._timestamps),
            "avg_frame_ms": float(deltas.mean()),
            "max_frame_ms": float(deltas.max()),
            "hitches": len(self._hitches),
            "hitch_details": self._hitches[-10:],  # last 10
        }


# Bug prioritizer                                                      

SEVERITY_LABELS = {
    range(1, 4): "low",
    range(4, 7): "medium",
    range(7, 9): "high",
    range(9, 11): "critical",
}


def severity_label(score: float) -> str:
    for r, label in SEVERITY_LABELS.items():
        if int(score) in r:
            return label
    return "critical" if score >= 10 else "low"


class BugPrioritizer:

    def score_bug(
        self,
        visual_mse: float,
        log_severity: int,
        frame_diff: float,
        hitch_ratio: float = 1.0,
    ) -> dict:
        # normalise each signal to [0, 10]
        visual_score = min(visual_mse / 0.1 * 10, 10)
        diff_score = min(frame_diff / 0.5 * 10, 10)
        hitch_score = min((hitch_ratio - 1) / 4 * 10, 10) if hitch_ratio > 1 else 0

        # weighted composite
        composite = (
            0.4 * visual_score
            + 0.35 * log_severity
            + 0.15 * diff_score
            + 0.10 * hitch_score
        )

        return {
            "composite_score": round(composite, 2),
            "severity": severity_label(composite),
            "components": {
                "visual_anomaly": round(visual_score, 2),
                "log_severity": log_severity,
                "frame_diff": round(diff_score, 2),
                "performance_hitch": round(hitch_score, 2),
            },
        }
