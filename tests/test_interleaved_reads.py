"""Regression tests: interleaved reads while a second instance writes.

Bug scenario: the data operator's web instance renders a transfer while
another instance finishes publishing the archive and then commits a late
parity frame. ``Storage.state()`` / ``list_deliveries()`` used to read the
delivery row, the frames and the parity through separate autocommit
connections, so the two commits could be spliced into a state that never
existed — the page showed pre-publish missing frames together with the
post-publish parity frame and even offered a parity recovery.

Fix: every logical read runs inside a single SQLite read transaction (a WAL
snapshot), so registration, received frames, parity, missing frames,
verification digest and publish status all come from the same committed
moment. A real transitional state may still be shown while a publication is
in flight; a spliced one may not.

The interleaving is driven deterministically: a hook on the reader's
connection factory lets the "other instance" commit exactly between the
reader's statements. With the fix, each logical read opens exactly one
connection, so the hook can only ever fire before or after a whole read.
"""

from __future__ import annotations

import hashlib
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import create_app  # noqa: E402
from app.storage import (  # noqa: E402
    FRAME_BYTES,
    STATUS_PUBLISHED,
    STATUS_WAITING,
    NotFoundError,
    Storage,
    xor_bytes,
)

TID = "STA-IL-001"


def make_case(count: int = 3) -> tuple[bytes, list[bytes], bytes, str]:
    size = count * FRAME_BYTES - 100
    archive = bytes((i * 37 + 11) % 256 for i in range(size))
    padded = archive + b"\x00" * (count * FRAME_BYTES - size)
    frames = [padded[i * FRAME_BYTES : (i + 1) * FRAME_BYTES] for i in range(count)]
    return archive, frames, xor_bytes(frames), hashlib.sha256(archive).hexdigest()


# Every state the delivery genuinely commits during the scenario, in order:
#   S0  waiting, frames 1-2, no parity
#   S1  waiting, frames 1-3, no parity  (frame commit just before the publish
#                                        commit: a real transitional state)
#   S2  published, frames 1-3, no parity
#   S3  published, frames 1-3, parity   (late parity after publication)
# Anything else — e.g. (waiting, 1-2, parity) or (waiting, 1-3, parity) —
# never existed in a single commit and must never be observed.
REAL_STATES = {
    (STATUS_WAITING, (1, 2), False, (3,)),
    (STATUS_WAITING, (1, 2, 3), False, ()),
    (STATUS_PUBLISHED, (1, 2, 3), False, ()),
    (STATUS_PUBLISHED, (1, 2, 3), True, ()),
}


def observed(state) -> tuple:
    return (
        state.status,
        state.received_frames,
        state.parity is not None,
        state.missing_frames,
    )


class Interleaver:
    """Let a second instance commit when the reader opens its Nth connection.

    ``fire_on = 0`` disarms the hook. While armed, connections opened by the
    reader instance are counted; when the count reaches ``fire_on`` the
    writer's commits are applied before that connection is used — exactly the
    mid-read interleaving two web instances produce under load.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch, reader: Storage, advance) -> None:
        self.fire_on = 0
        self.count = 0
        self.fired = False
        original = Storage._connect

        def patched(storage_self):
            conn = original(storage_self)
            if storage_self is reader and self.fire_on:
                self.count += 1
                if self.count == self.fire_on and not self.fired:
                    self.fired = True
                    advance()
            return conn

        monkeypatch.setattr(Storage, "_connect", patched)


@pytest.fixture
def scenario(tmp_path, monkeypatch):
    """Two instances on one data root; the reader's connections are hooked."""
    root = tmp_path / "data"
    archive, frames, parity, digest = make_case()
    writer = Storage(root)
    writer.register_delivery(TID, 3, len(archive), digest)
    writer.ingest_frames(TID, [(1, frames[0]), (2, frames[1])])  # -> S0

    def advance() -> None:
        # The other instance: the last frame arrives and publishes the
        # archive, then a late parity frame is committed on top.
        writer.ingest_frames(TID, [(3, frames[2])])
        writer.ingest_parity(TID, parity)

    reader = Storage(root)
    interleaver = Interleaver(monkeypatch, reader, advance)
    return SimpleNamespace(
        root=root,
        archive=archive,
        digest=digest,
        reader=reader,
        advance=advance,
        interleaver=interleaver,
    )


# --------------------------------------------------------------------- storage
@pytest.mark.parametrize("fire_on", [2, 3])
def test_state_read_is_one_committed_snapshot(scenario, fire_on):
    """A writer committing mid-read must not splice the observed state."""
    scenario.interleaver.fire_on = fire_on
    state = scenario.reader.state(TID)
    assert observed(state) in REAL_STATES

    # Commits that land before a read starts are seen as one later moment.
    scenario.advance()
    assert observed(scenario.reader.state(TID)) == (
        STATUS_PUBLISHED,
        (1, 2, 3),
        True,
        (),
    )


def test_ledger_records_are_each_one_committed_snapshot(scenario):
    """Every ledger record comes from one commit, even mid-list interleaving."""
    scenario.interleaver.fire_on = 3
    states = scenario.reader.list_deliveries()
    assert len(states) == 1
    assert observed(states[0]) in REAL_STATES

    scenario.advance()
    assert observed(scenario.reader.list_deliveries()[0]) == (
        STATUS_PUBLISHED,
        (1, 2, 3),
        True,
        (),
    )


def test_concurrent_reads_never_observe_spliced_state(tmp_path):
    """Threads reading while another instance publishes see only real states."""
    root = tmp_path / "data"
    archive, frames, parity, digest = make_case()
    writer = Storage(root)
    writer.register_delivery(TID, 3, len(archive), digest)
    writer.ingest_frames(TID, [(1, frames[0]), (2, frames[1])])

    stop = threading.Event()
    violations: list[tuple] = []

    def reader_loop() -> None:
        local = Storage(root)
        while not stop.is_set():
            obs = observed(local.state(TID))
            if obs not in REAL_STATES:
                violations.append(obs)
            for item in local.list_deliveries():
                if observed(item) not in REAL_STATES:
                    violations.append(observed(item))

    threads = [threading.Thread(target=reader_loop) for _ in range(4)]
    for thread in threads:
        thread.start()
    time.sleep(0.05)
    writer.ingest_frames(TID, [(3, frames[2])])  # publishes
    time.sleep(0.05)
    writer.ingest_parity(TID, parity)  # late parity
    time.sleep(0.05)
    stop.set()
    for thread in threads:
        thread.join(timeout=30)
    assert not violations


def test_download_during_interleaved_publish_serves_only_verified_archive(tmp_path):
    """Downloads stay fail-closed: 404 before publish, exact bytes after."""
    root = tmp_path / "data"
    archive, frames, parity, digest = make_case()
    writer = Storage(root)
    writer.register_delivery(TID, 3, len(archive), digest)
    writer.ingest_frames(TID, [(1, frames[0]), (2, frames[1])])

    reader = Storage(root)
    with pytest.raises(NotFoundError):
        reader.published_path(TID)

    writer.ingest_frames(TID, [(3, frames[2])])
    writer.ingest_parity(TID, parity)

    path, published_digest = reader.published_path(TID)
    assert path.read_bytes() == archive
    assert published_digest == digest


# ------------------------------------------------------------------------- web
@pytest.fixture
def two_instances(tmp_path, monkeypatch):
    """Two web instances (separate apps) interleaving on one data root."""
    from fastapi.testclient import TestClient

    root = tmp_path / "data"
    archive, frames, parity, digest = make_case()
    app_a = create_app(root)  # the data operator's instance (reader)
    app_b = create_app(root)  # the instance receiving frames (writer)
    writer = app_b.state.storage
    writer.register_delivery(TID, 3, len(archive), digest)
    writer.ingest_frames(TID, [(1, frames[0]), (2, frames[1])])  # -> S0

    def advance() -> None:
        writer.ingest_frames(TID, [(3, frames[2])])
        writer.ingest_parity(TID, parity)

    interleaver = Interleaver(monkeypatch, app_a.state.storage, advance)
    interleaver.fire_on = 3
    return SimpleNamespace(client=TestClient(app_a), advance=advance)


def test_detail_page_never_mixes_missing_frames_with_late_parity(two_instances):
    """The reported bug: published + late parity, page still showed a missing
    frame and offered a parity recovery — a state that never existed."""
    html = two_instances.client.get(f"/deliveries/{TID}").text
    # The read snapshot predates the other instance's commits, so the page
    # must show the pure pre-publish state: waiting, frame 3 missing, and —
    # crucially — no parity frame and no recovery offer.
    assert "等待补帧" in html
    assert "尚缺第 3 帧" in html
    assert "无校验帧" in html
    assert "可临时恢复" not in html
    assert "已核验发布" not in html

    # After the other instance's commits, the next render shows the complete
    # published state — again from a single committed moment.
    two_instances.advance()
    html = two_instances.client.get(f"/deliveries/{TID}").text
    assert "已核验发布" in html
    assert "尚缺第" not in html
    assert "可临时恢复" not in html


def test_api_detail_never_returns_spliced_state(two_instances):
    payload = two_instances.client.get(f"/api/deliveries/{TID}").json()
    assert payload["status"] == "waiting"
    assert payload["received_frames"] == [1, 2]
    assert payload["missing_frames"] == [3]
    assert payload["has_parity"] is False
    assert payload["parity_recoverable"] is False
    assert payload["published"] is None

    two_instances.advance()
    payload = two_instances.client.get(f"/api/deliveries/{TID}").json()
    assert payload["status"] == "published"
    assert payload["received_frames"] == [1, 2, 3]
    assert payload["missing_frames"] == []
    assert payload["has_parity"] is True
    assert payload["published"]["sha256"] is not None


def test_index_ledger_row_is_one_committed_snapshot(two_instances):
    html = two_instances.client.get("/").text
    assert "等待 1" in html  # one waiting record in the counts
    assert "有校验帧" not in html  # parity must not appear next to the gaps

    two_instances.advance()
    html = two_instances.client.get("/").text
    assert "已发布 1" in html
    assert "已核验发布" in html
