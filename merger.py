"""Merge the four test phases into one timeline CSV per episode.

Output files are ``episode1.csv``, ``episode2.csv`` and ``episode3.csv``.
Each output row represents one Unix timestamp (one second).  Access and error
logs can contain several events in the same second, so their parsed records are
kept losslessly in JSON columns.  Locust stats and Docker data are pivoted into
ordinary CSV columns; Docker column names include their container name.

Run from this directory:
    python merger.py
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PHASES = ("normal", "warning", "highvolume", "attack")
EPISODES = (1, 2, 3)

ACCESS_RE = re.compile(
    r"^(?P<remote_addr>\S+) \S+ \S+ \[(?P<time_local>[^]]+)\] "
    r'"(?P<request>[^\"]*)" (?P<status>\d{3}|-) (?P<body_bytes_sent>\S+) '
    r'"(?P<http_referer>[^\"]*)" "(?P<http_user_agent>[^\"]*)"'
)
ERROR_RE = re.compile(r"^(?P<time_local>\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}) \[(?P<level>[^]]+)] (?P<message>.*)$")


def unix_time(value: str, fmt: str) -> int:
    """Parse log timestamps as UTC and return second precision Unix time."""
    return int(datetime.strptime(value, fmt).replace(tzinfo=timezone.utc).timestamp())


def safe_name(value: str) -> str:
    """Make a stable, readable CSV-column suffix."""
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower() or "unknown"


def locate(kind: str, phase: str, episode: int) -> Path | None:
    """Return a matching source file, accepting either .log or .log.txt."""
    if kind == "access":
        choices = (f"access_{phase}{episode}.log.txt", f"access_{phase}{episode}.log")
    elif kind == "error":
        choices = (f"error_{phase}{episode}.log.txt", f"error_{phase}{episode}.log")
    elif kind == "stats":
        choices = (f"{phase}{episode}_stats_history.csv",)
    else:
        choices = (f"docker_stats_{phase}{episode}.csv",)
    return next((ROOT / name for name in choices if (ROOT / name).is_file()), None)


def add_access(rows: dict[int, dict[str, Any]], path: Path | None) -> None:
    if not path:
        return
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = ACCESS_RE.match(raw.strip())
        if not match:
            continue
        event = match.groupdict()
        event["raw"] = raw
        event["timestamp"] = unix_time(event["time_local"], "%d/%b/%Y:%H:%M:%S %z")
        request_parts = event["request"].split(" ", 2)
        event["method"] = request_parts[0] if request_parts else ""
        event["path"] = request_parts[1] if len(request_parts) > 1 else ""
        event["protocol"] = request_parts[2] if len(request_parts) > 2 else ""
        row = rows[event["timestamp"]]
        row["access_events"].append(event)
        row["access_request_count"] += 1
        row["access_status_counts"][event["status"]] += 1


def add_errors(rows: dict[int, dict[str, Any]], path: Path | None) -> None:
    if not path:
        return
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = ERROR_RE.match(raw.strip())
        if not match:
            continue
        event = match.groupdict()
        event["raw"] = raw
        event["timestamp"] = unix_time(event["time_local"], "%Y/%m/%d %H:%M:%S")
        row = rows[event["timestamp"]]
        row["error_events"].append(event)
        row["error_count"] += 1
        row["error_level_counts"][event["level"]] += 1


def add_stats(rows: dict[int, dict[str, Any]], path: Path | None) -> None:
    if not path:
        return
    with path.open(encoding="utf-8-sig", newline="") as source:
        for record in csv.DictReader(source):
            timestamp = int(float(record.pop("Timestamp")))
            label = safe_name("_".join(record.get(key, "") for key in ("Type", "Name")))
            for column, value in record.items():
                if column not in {"Type", "Name"}:
                    rows[timestamp][f"stats_{label}_{safe_name(column)}"] = value


def add_docker(rows: dict[int, dict[str, Any]], path: Path | None) -> None:
    if not path:
        return
    with path.open(encoding="utf-8-sig", newline="") as source:
        for record in csv.DictReader(source):
            timestamp = int(float(record.pop("Timestamp")))
            container = safe_name(record.pop("Container", "unknown"))
            for column, value in record.items():
                rows[timestamp][f"docker_{container}_{safe_name(column)}"] = value


def new_row(episode: int, phase: str) -> dict[str, Any]:
    return {
        "episode": episode, "phase": phase, "access_events": [], "error_events": [],
        "access_request_count": 0, "error_count": 0,
        "access_status_counts": Counter(), "error_level_counts": Counter(),
    }


def serialise(value: Any) -> str:
    if isinstance(value, (list, Counter)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=dict)
    return str(value)


def merge_episode(episode: int) -> None:
    rows: dict[int, dict[str, Any]] = defaultdict(lambda: new_row(episode, ""))
    for phase in PHASES:
        # Source phases do not overlap in the supplied data. If a future run does
        # overlap, the explicit phase stored in the source values remains visible.
        for kind, loader in (("access", add_access), ("error", add_errors), ("stats", add_stats), ("docker", add_docker)):
            before = set(rows)
            loader(rows, locate(kind, phase, episode))
            for timestamp in set(rows) - before:
                rows[timestamp]["episode"] = episode
                rows[timestamp]["phase"] = phase
        for row in rows.values():
            if not row["phase"]:
                row["phase"] = phase

    fixed = ["timestamp", "datetime_utc", "episode", "phase", "access_request_count", "access_status_counts", "access_events", "error_count", "error_level_counts", "error_events"]
    dynamic = sorted({key for row in rows.values() for key in row if key not in fixed})
    output = ROOT / f"episode{episode}.csv"
    with output.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fixed + dynamic, extrasaction="ignore")
        writer.writeheader()
        for timestamp in sorted(rows):
            row = rows[timestamp]
            row["timestamp"] = timestamp
            row["datetime_utc"] = datetime.fromtimestamp(timestamp, timezone.utc).isoformat()
            writer.writerow({key: serialise(row.get(key, "")) for key in fixed + dynamic})
    print(f"Wrote {output.name}: {len(rows):,} timestamp rows")


if __name__ == "__main__":
    for episode_number in EPISODES:
        merge_episode(episode_number)
