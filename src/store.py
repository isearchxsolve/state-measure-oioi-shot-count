"""SQLite source of truth for idempotent film/shot/clip-job state.

The database is deliberately independent from the legacy demo database.  A unique
``job_key`` is the durable credit-safety boundary: re-opening this database after a
process crash cannot insert a second generation for the same requested render.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .models import ShotSpec


def _now() -> float:
    return time.time()


def _dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _loads(value: Optional[str], default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


class FilmJobStore:
    """Transactional SQLite store for films and canonical render jobs."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    @contextmanager
    def _connection(self):
        """Close each SQLite handle promptly (important on Windows artifact cleanup)."""
        conn = self._connect()
        try:
            yield conn
        finally:
            conn.close()

    def _initialize(self) -> None:
        with self._lock, self._connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS films (
                    id TEXT PRIMARY KEY,
                    request_key TEXT NOT NULL UNIQUE,
                    script TEXT NOT NULL,
                    target_seconds INTEGER NOT NULL,
                    aspect_ratio TEXT NOT NULL,
                    bible_json TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL,
                    admission_mode TEXT NOT NULL,
                    capacity_at_admission INTEGER NOT NULL DEFAULT 0,
                    motion_clip_limit INTEGER NOT NULL DEFAULT 0,
                    logical_shots INTEGER NOT NULL DEFAULT 0,
                    unique_jobs INTEGER NOT NULL DEFAULT 0,
                    output_path TEXT,
                    last_error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS shots (
                    film_id TEXT NOT NULL REFERENCES films(id) ON DELETE CASCADE,
                    shot_index INTEGER NOT NULL,
                    canonical_index INTEGER NOT NULL,
                    prompt TEXT NOT NULL,
                    duration_seconds INTEGER NOT NULL,
                    image_refs_json TEXT NOT NULL,
                    seed INTEGER,
                    continuity_json TEXT NOT NULL,
                    hero_shot INTEGER NOT NULL DEFAULT 0,
                    job_key TEXT NOT NULL,
                    PRIMARY KEY (film_id, shot_index)
                );
                CREATE INDEX IF NOT EXISTS idx_shots_job_key ON shots(job_key);
                CREATE TABLE IF NOT EXISTS clip_jobs (
                    job_key TEXT PRIMARY KEY,
                    source_film_id TEXT NOT NULL REFERENCES films(id),
                    prompt TEXT NOT NULL,
                    duration_seconds INTEGER NOT NULL,
                    image_refs_json TEXT NOT NULL,
                    seed INTEGER,
                    aspect_ratio TEXT NOT NULL,
                    provider TEXT,
                    seat TEXT,
                    status TEXT NOT NULL,
                    remote_job_id TEXT,
                    remote_url TEXT,
                    local_path TEXT,
                    last_error TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    next_attempt_at REAL NOT NULL DEFAULT 0,
                    last_polled_at REAL NOT NULL DEFAULT 0,
                    credits_reserved REAL NOT NULL DEFAULT 0,
                    credits_spent REAL NOT NULL DEFAULT 0,
                    is_fallback INTEGER NOT NULL DEFAULT 0,
                    fallback_reason TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_clip_jobs_schedule ON clip_jobs(status, next_attempt_at);
                CREATE TABLE IF NOT EXISTS unavailable_seats (
                    seat TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (seat, provider)
                );
                CREATE TABLE IF NOT EXISTS assembly_segments (
                    film_id TEXT NOT NULL REFERENCES films(id) ON DELETE CASCADE,
                    segment_index INTEGER NOT NULL,
                    start_shot_index INTEGER NOT NULL,
                    end_shot_index INTEGER NOT NULL,
                    path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (film_id, segment_index)
                );
                """
            )
            # The project is intentionally migration-light for the local durable SQLite
            # store.  Existing film databases must remain resumable after an upgrade.
            columns = {str(row["name"]) for row in conn.execute("PRAGMA table_info(shots)").fetchall()}
            if "hero_shot" not in columns:
                conn.execute("ALTER TABLE shots ADD COLUMN hero_shot INTEGER NOT NULL DEFAULT 0")
            job_columns = {str(row["name"]) for row in conn.execute("PRAGMA table_info(clip_jobs)").fetchall()}
            if "is_fallback" not in job_columns:
                conn.execute("ALTER TABLE clip_jobs ADD COLUMN is_fallback INTEGER NOT NULL DEFAULT 0")
            if "fallback_reason" not in job_columns:
                conn.execute("ALTER TABLE clip_jobs ADD COLUMN fallback_reason TEXT")
            film_columns = {str(row["name"]) for row in conn.execute("PRAGMA table_info(films)").fetchall()}
            if "motion_clip_limit" not in film_columns:
                conn.execute("ALTER TABLE films ADD COLUMN motion_clip_limit INTEGER NOT NULL DEFAULT 0")

    @staticmethod
    def _film(row: sqlite3.Row) -> Dict[str, Any]:
        result = dict(row)
        result["bible"] = _loads(result.pop("bible_json", "{}"), {})
        return result

    @staticmethod
    def _job(row: sqlite3.Row) -> Dict[str, Any]:
        result = dict(row)
        result["image_refs"] = _loads(result.pop("image_refs_json", "[]"), [])
        return result

    @staticmethod
    def _shot(row: sqlite3.Row) -> Dict[str, Any]:
        result = dict(row)
        result["image_refs"] = _loads(result.pop("image_refs_json", "[]"), [])
        result["continuity"] = _loads(result.pop("continuity_json", "{}"), {})
        return result

    def get_film_by_request_key(self, request_key: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._connection() as conn:
            row = conn.execute("SELECT * FROM films WHERE request_key=?", (request_key,)).fetchone()
        return self._film(row) if row else None

    def create_film(
        self,
        *,
        request_key: str,
        script: str,
        target_seconds: int,
        aspect_ratio: str,
        bible: Dict[str, Any],
        status: str,
        admission_mode: str,
        capacity_at_admission: int,
        motion_clip_limit: int = 0,
    ) -> Dict[str, Any]:
        """Create or return the existing film for the exact deterministic request."""
        now = _now()
        film_id = f"film_{uuid.uuid4().hex}"
        with self._lock, self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """INSERT INTO films (id,request_key,script,target_seconds,aspect_ratio,bible_json,status,
                   admission_mode,capacity_at_admission,motion_clip_limit,created_at,updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(request_key) DO NOTHING""",
                (film_id, request_key, script, int(target_seconds), aspect_ratio, _dumps(bible), status,
                 admission_mode, int(capacity_at_admission), max(0, int(motion_clip_limit)), now, now),
            )
            row = conn.execute("SELECT * FROM films WHERE request_key=?", (request_key,)).fetchone()
            conn.commit()
        return self._film(row)

    def get_film(self, film_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._connection() as conn:
            row = conn.execute("SELECT * FROM films WHERE id=?", (film_id,)).fetchone()
        return self._film(row) if row else None

    def set_film_status(self, film_id: str, status: str, *, error: Optional[str] = None, output_path: Optional[str] = None) -> None:
        with self._lock, self._connection() as conn:
            conn.execute(
                "UPDATE films SET status=?, last_error=?, output_path=COALESCE(?,output_path), updated_at=? WHERE id=?",
                (status, error, output_path, _now(), film_id),
            )

    def add_shots_and_jobs(self, film_id: str, shots: Iterable[ShotSpec], *, aspect_ratio: str, job_keys: Dict[int, str]) -> None:
        """Persist every logical shot and insert each canonical job once atomically."""
        materialized = list(shots)
        now = _now()
        canonical: Dict[int, ShotSpec] = {}
        for shot in materialized:
            canonical.setdefault(shot.canonical_index if shot.canonical_index is not None else shot.index, shot)
        with self._lock, self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            for shot in materialized:
                canon = shot.canonical_index if shot.canonical_index is not None else shot.index
                conn.execute(
                    """INSERT OR IGNORE INTO shots
                       (film_id,shot_index,canonical_index,prompt,duration_seconds,image_refs_json,seed,continuity_json,hero_shot,job_key)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (film_id, shot.index, canon, shot.prompt, int(shot.duration_seconds), _dumps(shot.image_refs),
                     shot.seed, _dumps(shot.continuity), int(bool(shot.hero_shot)), job_keys[canon]),
                )
            for canon, shot in canonical.items():
                conn.execute(
                    """INSERT INTO clip_jobs
                       (job_key,source_film_id,prompt,duration_seconds,image_refs_json,seed,aspect_ratio,status,created_at,updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?) ON CONFLICT(job_key) DO NOTHING""",
                    (job_keys[canon], film_id, shot.prompt, int(shot.duration_seconds), _dumps(shot.image_refs),
                     shot.seed, aspect_ratio, "pending", now, now),
                )
            conn.execute(
                "UPDATE films SET logical_shots=?, unique_jobs=?, updated_at=? WHERE id=?",
                (len(materialized), len(canonical), now, film_id),
            )
            conn.commit()

    def shots(self, film_id: str) -> List[Dict[str, Any]]:
        with self._lock, self._connection() as conn:
            rows = conn.execute("SELECT * FROM shots WHERE film_id=? ORDER BY shot_index", (film_id,)).fetchall()
        return [self._shot(row) for row in rows]

    def shot_for_job(self, film_id: str, job_key: str) -> Optional[Dict[str, Any]]:
        """Return the first timeline slot for a canonical job.

        A canonical job can serve several identical logical shots.  The first one owns
        the reference image and continuity metadata used for a keyless fallback; all
        later slots reuse that same finished local MP4 during ordered assembly.
        """
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM shots WHERE film_id=? AND job_key=? ORDER BY shot_index LIMIT 1",
                (film_id, job_key),
            ).fetchone()
        return self._shot(row) if row else None

    def jobs_for_film(self, film_id: str) -> List[Dict[str, Any]]:
        with self._lock, self._connection() as conn:
            rows = conn.execute(
                """SELECT DISTINCT j.* FROM clip_jobs j JOIN shots s ON s.job_key=j.job_key
                   WHERE s.film_id=? ORDER BY j.created_at,j.job_key""",
                (film_id,),
            ).fetchall()
        return [self._job(row) for row in rows]

    def get_job(self, job_key: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._connection() as conn:
            row = conn.execute("SELECT * FROM clip_jobs WHERE job_key=?", (job_key,)).fetchone()
        return self._job(row) if row else None

    def due_jobs(self, film_id: str, *, now: Optional[float] = None, limit: int = 100) -> List[Dict[str, Any]]:
        now = _now() if now is None else now
        with self._lock, self._connection() as conn:
            rows = conn.execute(
                """SELECT DISTINCT j.* FROM clip_jobs j JOIN shots s ON s.job_key=j.job_key
                   WHERE s.film_id=? AND j.status IN ('pending','retry_wait','parked_capacity')
                      AND j.next_attempt_at<=?
                    ORDER BY EXISTS(SELECT 1 FROM shots hero WHERE hero.film_id=s.film_id
                                    AND hero.job_key=j.job_key AND hero.hero_shot=1) DESC,
                             j.next_attempt_at,j.created_at LIMIT ?""",
                (film_id, now, int(limit)),
            ).fetchall()
        return [self._job(row) for row in rows]

    def claim_for_submit(self, job_key: str, *, provider: str, seat: Optional[str], now: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Atomically give one scheduler tick the exclusive right to POST this job.

        ``submitting`` is durable before the network request.  A concurrent tick observes
        that state and cannot buy a second clip for the same stable job key.
        """
        now = _now() if now is None else now
        with self._lock, self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            changed = conn.execute(
                """UPDATE clip_jobs SET status='submitting',provider=?,seat=?,attempts=attempts+1,
                   updated_at=? WHERE job_key=? AND status IN ('pending','retry_wait','parked_capacity')
                   AND next_attempt_at<=?""",
                (provider, seat, now, job_key, now),
            ).rowcount
            row = conn.execute("SELECT * FROM clip_jobs WHERE job_key=?", (job_key,)).fetchone() if changed else None
            conn.commit()
        return self._job(row) if row else None

    def recover_stale_claims(self, film_id: str, *, age_seconds: float = 90.0) -> int:
        """Release a claim left behind by a process crash before/after ``POST``.

        Retrying the exact durable ``job_key`` is safe under the frozen router's
        idempotency rule.  The conservative age avoids racing an ordinary slow HTTP
        request while making a dead worker resumable without manual database edits.
        """
        cutoff = _now() - max(5.0, float(age_seconds))
        with self._lock, self._connection() as conn:
            changed = conn.execute(
                """UPDATE clip_jobs SET status='retry_wait',next_attempt_at=?,
                   last_error=COALESCE(last_error,'recovered stale submit claim'),updated_at=?
                   WHERE source_film_id=? AND status='submitting' AND updated_at<=?""",
                (_now(), _now(), film_id, cutoff),
            ).rowcount
        return int(changed)

    def due_polls(self, film_id: str, *, poll_interval_seconds: float = 5.0, now: Optional[float] = None) -> List[Dict[str, Any]]:
        now = _now() if now is None else now
        with self._lock, self._connection() as conn:
            rows = conn.execute(
                """SELECT DISTINCT j.* FROM clip_jobs j JOIN shots s ON s.job_key=j.job_key
                   WHERE s.film_id=? AND j.status IN ('queued','running') AND j.remote_job_id IS NOT NULL
                     AND j.last_polled_at<=?
                   ORDER BY j.last_polled_at,j.created_at""",
                (film_id, now - max(5.0, poll_interval_seconds)),
            ).fetchall()
        return [self._job(row) for row in rows]

    def mark_submitted(self, job_key: str, *, remote_job_id: str, provider: str, seat: Optional[str], credits_reserved: float) -> None:
        with self._lock, self._connection() as conn:
            conn.execute(
                """UPDATE clip_jobs SET status='queued',remote_job_id=?,provider=?,seat=?,credits_reserved=?,
                   next_attempt_at=0,updated_at=? WHERE job_key=? AND status='submitting'""",
                (remote_job_id, provider, seat, float(credits_reserved), _now(), job_key),
            )

    def record_attempt(self, job_key: str) -> int:
        """Count a submitted/transport attempt without creating another render job."""
        with self._lock, self._connection() as conn:
            conn.execute("UPDATE clip_jobs SET attempts=attempts+1,updated_at=? WHERE job_key=?", (_now(), job_key))
            row = conn.execute("SELECT attempts FROM clip_jobs WHERE job_key=?", (job_key,)).fetchone()
        return int(row["attempts"] if row else 0)

    def note_poll_transport_error(self, job_key: str, error: str) -> None:
        """Record a poll-network problem without charging it to render attempts.

        The remote job may still be queued or running.  Its durable state remains queued/
        running and the normal poll interval gates the next GET.
        """
        with self._lock, self._connection() as conn:
            conn.execute(
                "UPDATE clip_jobs SET last_error=?,last_polled_at=?,updated_at=? WHERE job_key=?",
                (error[:1000], _now(), _now(), job_key),
            )

    def mark_fallback_pending(self, job_key: str, reason: str) -> None:
        """Convert an unavailable motion slot into a local keyless fallback slot."""
        with self._lock, self._connection() as conn:
            conn.execute(
                """UPDATE clip_jobs SET status='fallback_pending',is_fallback=1,fallback_reason=?,
                   last_error=?,next_attempt_at=0,updated_at=? WHERE job_key=?""",
                (reason[:1000], reason[:1000], _now(), job_key),
            )

    def fallback_jobs(self, film_id: str, *, limit: int = 1000) -> List[Dict[str, Any]]:
        with self._lock, self._connection() as conn:
            rows = conn.execute(
                """SELECT DISTINCT j.* FROM clip_jobs j JOIN shots s ON s.job_key=j.job_key
                   WHERE s.film_id=? AND j.status='fallback_pending' AND j.next_attempt_at<=?
                   ORDER BY EXISTS(SELECT 1 FROM shots hero WHERE hero.film_id=s.film_id
                                   AND hero.job_key=j.job_key AND hero.hero_shot=1) DESC,
                            j.created_at LIMIT ?""",
                (film_id, _now(), int(limit)),
            ).fetchall()
        return [self._job(row) for row in rows]

    def mark_fallback_success(self, job_key: str, local_path: str) -> None:
        with self._lock, self._connection() as conn:
            conn.execute(
                """UPDATE clip_jobs SET status='succeeded',provider='kenburns',local_path=?,
                   is_fallback=1,last_error=NULL,updated_at=? WHERE job_key=?""",
                (local_path, _now(), job_key),
            )

    def mark_poll(self, job_key: str, *, status: str, provider: str, url: Optional[str], error: Optional[str], credits_spent: float) -> None:
        with self._lock, self._connection() as conn:
            conn.execute(
                """UPDATE clip_jobs SET status=?, provider=?, remote_url=COALESCE(?,remote_url), last_error=?,
                   credits_spent=MAX(credits_spent,?),last_polled_at=?,updated_at=? WHERE job_key=?""",
                (status, provider, url, error, float(credits_spent), _now(), _now(), job_key),
            )

    def mark_local_success(self, job_key: str, local_path: str) -> None:
        with self._lock, self._connection() as conn:
            conn.execute(
                "UPDATE clip_jobs SET status='succeeded',local_path=?,last_error=NULL,updated_at=? WHERE job_key=?",
                (local_path, _now(), job_key),
            )

    def jobs_needing_materialization(self, film_id: str) -> List[Dict[str, Any]]:
        with self._lock, self._connection() as conn:
            rows = conn.execute(
                """SELECT DISTINCT j.* FROM clip_jobs j JOIN shots s ON s.job_key=j.job_key
                   WHERE s.film_id=? AND j.status='succeeded' AND (j.local_path IS NULL OR j.local_path='')
                   ORDER BY j.updated_at""",
                (film_id,),
            ).fetchall()
        return [self._job(row) for row in rows]

    def defer(self, job_key: str, *, status: str, seconds: float, error: str) -> None:
        with self._lock, self._connection() as conn:
            conn.execute(
                "UPDATE clip_jobs SET status=?,last_error=?,next_attempt_at=?,updated_at=? WHERE job_key=?",
                (status, error[:1000], _now() + max(0.0, seconds), _now(), job_key),
            )

    def quarantine(self, job_key: str, error: str) -> None:
        self.defer(job_key, status="quarantined", seconds=0, error=error)

    def terminal_failures(self, film_id: str) -> List[Dict[str, Any]]:
        """Jobs that cannot progress or assemble without a deliberate recovery action."""
        with self._lock, self._connection() as conn:
            rows = conn.execute(
                """SELECT DISTINCT j.* FROM clip_jobs j JOIN shots s ON s.job_key=j.job_key
                   WHERE s.film_id=? AND j.status IN ('quarantined','parked_wall')
                   ORDER BY j.updated_at,j.job_key""",
                (film_id,),
            ).fetchall()
        return [self._job(row) for row in rows]

    def remote_submission_count(self, film_id: str) -> int:
        """Number of durable motion jobs that have crossed the router boundary.

        This survives an API/worker restart, unlike a client-process counter, and is
        therefore the final safety gate for a per-film live-motion allowance.
        """
        with self._lock, self._connection() as conn:
            row = conn.execute(
                """SELECT COUNT(DISTINCT j.job_key) AS n FROM clip_jobs j
                   JOIN shots s ON s.job_key=j.job_key
                   WHERE s.film_id=? AND j.remote_job_id IS NOT NULL""",
                (film_id,),
            ).fetchone()
        return int(row["n"] if row else 0)

    def mark_seat_unavailable(self, seat: str, provider: str, reason: str) -> None:
        if not seat:
            return
        with self._lock, self._connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO unavailable_seats(seat,provider,reason,created_at) VALUES (?,?,?,?)",
                (seat, provider or "unknown", reason[:1000], _now()),
            )

    def unavailable_seats(self) -> set[tuple[str, str]]:
        with self._lock, self._connection() as conn:
            rows = conn.execute("SELECT seat,provider FROM unavailable_seats").fetchall()
        return {(str(row["seat"]), str(row["provider"])) for row in rows}

    def jobs_using_seat(self, provider: str, seat: str) -> int:
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM clip_jobs WHERE provider=? AND seat=? AND status IN ('submitting','queued','running')",
                (provider, seat),
            ).fetchone()
        return int(row["n"] if row else 0)

    def provider_inflight(self, provider: str) -> int:
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM clip_jobs WHERE provider=? AND status IN ('submitting','queued','running')", (provider,)
            ).fetchone()
        return int(row["n"] if row else 0)

    def save_segment(self, film_id: str, *, segment_index: int, start_shot_index: int, end_shot_index: int, path: str, status: str = "succeeded") -> None:
        with self._lock, self._connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO assembly_segments
                   (film_id,segment_index,start_shot_index,end_shot_index,path,status,created_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (film_id, segment_index, start_shot_index, end_shot_index, path, status, _now()),
            )

    def segments(self, film_id: str) -> List[Dict[str, Any]]:
        with self._lock, self._connection() as conn:
            rows = conn.execute("SELECT * FROM assembly_segments WHERE film_id=? ORDER BY segment_index", (film_id,)).fetchall()
        return [dict(row) for row in rows]

    def costs(self, film_id: str) -> Dict[str, Any]:
        """Spend/reservations by film/provider/seat plus a transparent projection."""
        jobs = self.jobs_for_film(film_id)
        by_provider: Dict[str, Dict[str, float]] = {}
        by_seat: Dict[str, Dict[str, float]] = {}
        spent = reserved = 0.0
        completed = 0
        unit_costs: List[float] = []
        for job in jobs:
            provider = job.get("provider") or "unassigned"
            seat = job.get("seat") or "unassigned"
            j_spent = float(job.get("credits_spent") or 0)
            j_reserved = float(job.get("credits_reserved") or 0)
            spent += j_spent
            reserved += j_reserved
            bucket = by_provider.setdefault(provider, {"spent": 0.0, "reserved": 0.0, "jobs": 0.0})
            bucket["spent"] += j_spent
            bucket["reserved"] += j_reserved
            bucket["jobs"] += 1
            sb = by_seat.setdefault(f"{provider}/{seat}", {"spent": 0.0, "reserved": 0.0, "jobs": 0.0})
            sb["spent"] += j_spent
            sb["reserved"] += j_reserved
            sb["jobs"] += 1
            if job["status"] == "succeeded" and (j_spent or j_reserved):
                completed += 1
                unit_costs.append(j_spent or j_reserved)
        average = (sum(unit_costs) / len(unit_costs)) if unit_costs else 30.0
        remaining = sum(1 for job in jobs if job["status"] not in {"succeeded", "quarantined", "parked_wall"})
        return {
            "spent": round(spent, 2), "reserved": round(reserved, 2),
            "projected_total": round(spent + remaining * average, 2),
            "by_provider": by_provider, "by_seat": by_seat,
            "unique_jobs": len(jobs), "completed_jobs": completed,
            "fallback_jobs": sum(1 for job in jobs if bool(job.get("is_fallback"))),
        }

    def summary(self, film_id: str) -> Dict[str, Any]:
        film = self.get_film(film_id)
        if not film:
            raise KeyError(f"film {film_id} does not exist")
        jobs = self.jobs_for_film(film_id)
        states: Dict[str, int] = {}
        for job in jobs:
            states[job["status"]] = states.get(job["status"], 0) + 1
        return {**film, "job_states": states, "cost": self.costs(film_id), "segments": self.segments(film_id)}
