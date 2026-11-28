"""Shared-filesystem atomic task lease manager for Spot-safe Stage-1 workers.

Coordinates multiple independent single-GPU and single-node DDP workers over a shared
POSIX directory (e.g. shared NFS/Lustre mount) without external databases or persisted
host/pod identifiers.
"""
from __future__ import annotations

import errno
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from ..core.io import sha256_file
from .spec import RunSpec, assert_no_forbidden_infra_strings

LEASE_SCHEMA = "plm_parametric_capacity_lease_v1"
COMPLETED_STATUSES = frozenset({"completed", "complete"})
DEFAULT_LEASE_TTL_SEC: float = 600.0
DEFAULT_HEARTBEAT_INTERVAL_SEC: float = 60.0
DEFAULT_STEAL_LOCK_TTL_SEC: float = 60.0


def generate_worker_token(prefix: str = "spot") -> str:
    """Return a clean opaque worker identifier containing zero host/pod/IP tokens."""
    token = f"{prefix}_{uuid.uuid4().hex[:12]}"
    assert_no_forbidden_infra_strings(token, label="worker_token")
    return token


def is_run_receipt_complete(run_json_path: Path) -> bool:
    """Verify receipt exists, passes optional SHA-256 sidecar, and has completed status."""
    if not run_json_path.is_file():
        return False
    sidecar = run_json_path.with_name(run_json_path.name + ".sha256")
    if sidecar.is_file():
        parts = sidecar.read_text(encoding="utf-8").strip().split()
        if not parts or parts[0] != sha256_file(run_json_path):
            return False
    try:
        with run_json_path.open("r", encoding="utf-8") as handle:
            doc = json.load(handle)
    except Exception:
        return False
    return isinstance(doc, dict) and doc.get("status") in COMPLETED_STATUSES


def load_run_manifest(manifest_path: Path | str) -> list[RunSpec]:
    """Load a Stage-1 JSON manifest (either {'runs': [...]} or a top-level list of RunSpec dicts)."""
    path = Path(manifest_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if isinstance(raw, dict):
        items = raw.get("runs")
        if not isinstance(items, list):
            raise ValueError(f"Manifest dict {path} must contain a 'runs' list")
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError(f"Manifest {path} must be a JSON list or dict with 'runs'")

    specs: list[RunSpec] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"Manifest entry must be an object, got {type(item).__name__}")
        spec = RunSpec.from_dict(item)
        if spec.run_id in seen:
            raise ValueError(f"Duplicate run_id {spec.run_id!r} in manifest {path}")
        seen.add(spec.run_id)
        specs.append(spec)
    return specs


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    assert_no_forbidden_infra_strings(payload, label=path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    try:
        with tmp.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _create_exclusive_json(path: Path, payload: dict[str, Any]) -> bool:
    """Atomically create a new JSON file using O_CREAT | O_EXCL. Returns False if it exists."""
    assert_no_forbidden_infra_strings(payload, label=path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(str(path), flags, 0o644)
    except FileExistsError:
        return False
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            return False
        raise
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    return True


@dataclass(frozen=True)
class LeaseRecord:
    """Active lease claim for one run_id on shared storage."""

    run_id: str
    worker_id: str
    mode: str
    claimed_epoch_sec: float
    heartbeat_epoch_sec: float
    lease_ttl_sec: float
    path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": LEASE_SCHEMA,
            "run_id": self.run_id,
            "worker_id": self.worker_id,
            "mode": self.mode,
            "claimed_epoch_sec": float(self.claimed_epoch_sec),
            "heartbeat_epoch_sec": float(self.heartbeat_epoch_sec),
            "lease_ttl_sec": float(self.lease_ttl_sec),
        }


class SharedFSLeaseManager:
    """Atomic lease claim / heartbeat / expiration manager over a shared directory."""

    def __init__(
        self,
        out_dir: Path | str,
        *,
        lease_ttl_sec: float = DEFAULT_LEASE_TTL_SEC,
        steal_lock_ttl_sec: float = DEFAULT_STEAL_LOCK_TTL_SEC,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if lease_ttl_sec <= 0.0:
            raise ValueError(f"lease_ttl_sec must be > 0, got {lease_ttl_sec}")
        self.out_dir = Path(out_dir)
        self.runs_dir = self.out_dir / "runs"
        self.leases_dir = self.out_dir / "leases"
        self.lease_ttl_sec = float(lease_ttl_sec)
        self.steal_lock_ttl_sec = float(steal_lock_ttl_sec)
        self._clock = clock if clock is not None else time.time

    def lease_path(self, run_id: str) -> Path:
        return self.leases_dir / f"{run_id}.lease.json"

    def _steal_lock_path(self, run_id: str) -> Path:
        return self.leases_dir / f".{run_id}.steal.lock"

    def read_lease(self, run_id: str) -> dict[str, Any] | None:
        path = self.lease_path(run_id)
        if not path.is_file():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                doc = json.load(handle)
            return doc if isinstance(doc, dict) else None
        except (OSError, json.JSONDecodeError):
            return None

    def is_lease_stale(self, lease_doc: dict[str, Any], *, now: float | None = None) -> bool:
        cur = float(self._clock() if now is None else now)
        hb = float(lease_doc.get("heartbeat_epoch_sec", 0.0))
        ttl = float(lease_doc.get("lease_ttl_sec", self.lease_ttl_sec))
        eff_ttl = min(self.lease_ttl_sec, ttl) if ttl > 0.0 else self.lease_ttl_sec
        return (cur - hb) > eff_ttl

    def _acquire_steal_lock(self, run_id: str, worker_id: str, now: float) -> bool:
        lock_path = self._steal_lock_path(run_id)
        payload = {"worker_id": worker_id, "created_epoch_sec": now}
        if _create_exclusive_json(lock_path, payload):
            return True
        # Clean up abandoned steal lock if older than steal_lock_ttl_sec
        try:
            with lock_path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
            created = float(existing.get("created_epoch_sec", 0.0))
        except Exception:
            try:
                created = lock_path.stat().st_mtime
            except OSError:
                return False
        if (now - created) > self.steal_lock_ttl_sec:
            try:
                lock_path.unlink()
            except OSError:
                return False
            return _create_exclusive_json(lock_path, payload)
        return False

    def _release_steal_lock(self, run_id: str) -> None:
        lock_path = self._steal_lock_path(run_id)
        try:
            lock_path.unlink()
        except OSError:
            pass

    def try_claim(
        self,
        run_id: str,
        *,
        worker_id: str,
        mode: str = "single_gpu",
    ) -> LeaseRecord | None:
        """Atomically claim ``run_id`` if not completed and not held by a live lease."""
        assert_no_forbidden_infra_strings(worker_id, label="worker_id")
        if is_run_receipt_complete(self.runs_dir / f"{run_id}.json"):
            return None

        self.leases_dir.mkdir(parents=True, exist_ok=True)
        now = float(self._clock())
        path = self.lease_path(run_id)
        record = LeaseRecord(
            run_id=run_id,
            worker_id=worker_id,
            mode=mode,
            claimed_epoch_sec=now,
            heartbeat_epoch_sec=now,
            lease_ttl_sec=self.lease_ttl_sec,
            path=path,
        )

        # Fast path: atomic exclusive creation when no lease file exists
        if _create_exclusive_json(path, record.to_dict()):
            return record

        # Contested or stale lease path
        existing = self.read_lease(run_id)
        if existing is not None and not self.is_lease_stale(existing, now=now):
            return None

        if not self._acquire_steal_lock(run_id, worker_id=worker_id, now=now):
            return None
        try:
            # Re-check completion and staleness under steal lock
            if is_run_receipt_complete(self.runs_dir / f"{run_id}.json"):
                return None
            existing_recheck = self.read_lease(run_id)
            if existing_recheck is not None and not self.is_lease_stale(
                existing_recheck, now=float(self._clock())
            ):
                return None
            now2 = float(self._clock())
            stolen = LeaseRecord(
                run_id=run_id,
                worker_id=worker_id,
                mode=mode,
                claimed_epoch_sec=now2,
                heartbeat_epoch_sec=now2,
                lease_ttl_sec=self.lease_ttl_sec,
                path=path,
            )
            _atomic_write_json(path, stolen.to_dict())
            return stolen
        finally:
            self._release_steal_lock(run_id)

    def claim_next(
        self,
        specs: Sequence[RunSpec],
        *,
        worker_id: str,
        mode: str = "single_gpu",
        predicate: Callable[[RunSpec], bool] | None = None,
    ) -> tuple[RunSpec, LeaseRecord] | None:
        """Scan ``specs`` in order and atomically claim the first eligible unfinished run."""
        for spec in specs:
            if predicate is not None and not predicate(spec):
                continue
            lease = self.try_claim(spec.run_id, worker_id=worker_id, mode=mode)
            if lease is not None:
                return spec, lease
        return None

    def renew(self, lease: LeaseRecord) -> bool:
        """Refresh heartbeat timestamp if ``lease.worker_id`` still owns the lease."""
        existing = self.read_lease(lease.run_id)
        if existing is None or existing.get("worker_id") != lease.worker_id:
            return False
        now = float(self._clock())
        updated = dict(existing)
        updated["heartbeat_epoch_sec"] = now
        updated["lease_ttl_sec"] = float(self.lease_ttl_sec)
        _atomic_write_json(lease.path, updated)
        return True

    def release(self, lease: LeaseRecord) -> bool:
        """Release the lease file if it is still owned by ``lease.worker_id``."""
        existing = self.read_lease(lease.run_id)
        if existing is None or existing.get("worker_id") != lease.worker_id:
            return False
        try:
            lease.path.unlink()
            return True
        except OSError:
            return False


class LeaseHeartbeat:
    """Background daemon thread that periodically renews an active shared-FS lease."""

    def __init__(
        self,
        manager: SharedFSLeaseManager,
        lease: LeaseRecord,
        *,
        interval_sec: float = DEFAULT_HEARTBEAT_INTERVAL_SEC,
    ) -> None:
        self._manager = manager
        self._lease = lease
        self._interval = max(0.5, float(interval_sec))
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"lease-hb-{lease.run_id[:24]}",
            daemon=True,
        )

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval):
            try:
                if not self._manager.renew(self._lease):
                    return
            except Exception:
                # Transient shared-FS glitches should not kill the training worker;
                # retry on next interval.
                continue

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=5.0)

    def __enter__(self) -> LeaseHeartbeat:
        self.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.stop()
