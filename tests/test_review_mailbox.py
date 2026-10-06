"""Review finding F5: Mailbox.verdicts() must tolerate a half-written last line of verdicts.jsonl.

The solver appends to verdicts.jsonl while the EA polls it; an unsynchronised reader can see the last line partially
written. Before the fix json.loads raised ValueError, the EA exited 1 and pair.py stopped the whole run."""
from __future__ import annotations

import json
import threading
import time

from itrea.exchange.mailbox import Mailbox


def _mb(tmp_path, name="mb"):
    return Mailbox(tmp_path / name, "runner", create=True, run_id=name)


def test_missing_file_gives_empty(tmp_path):
    assert _mb(tmp_path).verdicts() == []


def test_partial_last_line_is_ignored_then_picked_up(tmp_path):
    mb = _mb(tmp_path)
    for s in range(3):
        mb.verdict(s, float(s), s % 2 == 0, note=json.dumps({"row": s}))
    line = json.dumps({"seq": 3, "t": 1.0, "obj": 3.0, "accepted": True, "note": ""}) + "\n"
    half = len(line) // 2
    with open(mb.dir / "verdicts.jsonl", "a") as f:
        f.write(line[:half])                     # what a reader can see mid-append: no newline yet
    before = mb.verdicts()                       # must not raise
    assert [v["seq"] for v in before] == [0, 1, 2]

    # Engine._verdicts slices by the previous length; the completed line must arrive exactly once at the next poll
    n_seen = len(before)
    with open(mb.dir / "verdicts.jsonl", "a") as f:
        f.write(line[half:])
    after = mb.verdicts()
    assert after[:3] == before
    assert [v["seq"] for v in after[n_seen:]] == [3]


def test_unterminated_but_parseable_tail_is_not_returned(tmp_path):
    # a tail that happens to parse (e.g. cut right after a closing brace of a shorter object) is still incomplete
    mb = _mb(tmp_path)
    mb.verdict(0, 1.0, True)
    with open(mb.dir / "verdicts.jsonl", "a") as f:
        f.write('{"seq": 1}')
    assert [v["seq"] for v in mb.verdicts()] == [0]


def test_malformed_complete_line_stops_parsing(tmp_path):
    # prefix semantics (as trace.read_events): never skip a line and resume, or a length-sliced reader would misalign
    mb = _mb(tmp_path)
    mb.verdict(0, 1.0, True)
    with open(mb.dir / "verdicts.jsonl", "a") as f:
        f.write("{not json\n")
    mb.verdict(2, 3.0, False)
    assert [v["seq"] for v in mb.verdicts()] == [0]


def test_blank_lines_and_nonfinite_obj(tmp_path):
    mb = _mb(tmp_path)
    mb.verdict(0, float("inf"), False)
    with open(mb.dir / "verdicts.jsonl", "a") as f:
        f.write("\n")
    mb.verdict(1, 2.0, True)
    vs = mb.verdicts()
    assert [v["seq"] for v in vs] == [0, 1]
    assert vs[0]["obj"] == float("inf")


def test_concurrent_append_reader_never_raises_and_only_grows(tmp_path):
    # Holds whether or not the race fires on this run: no exception, and each poll is a prefix of the next.
    mb = _mb(tmp_path)
    reader = Mailbox(tmp_path / "mb", "ea", run_id="mb")
    stop = threading.Event()
    n_total = 500

    def write():
        for s in range(n_total):
            mb.verdict(s, float(s), True, note=json.dumps({"row": s, "pad": "x" * 200}))
        stop.set()

    th = threading.Thread(target=write)
    th.start()
    prev: list[dict] = []
    deadline = time.monotonic() + 5.0
    try:
        while not stop.is_set() and time.monotonic() < deadline:
            cur = reader.verdicts()
            assert len(cur) >= len(prev)
            assert [v["seq"] for v in cur[:len(prev)]] == [v["seq"] for v in prev]
            prev = cur
            time.sleep(0.0002)                   # let the writer thread take the GIL
    finally:
        th.join()
    assert [v["seq"] for v in reader.verdicts()] == list(range(n_total))
