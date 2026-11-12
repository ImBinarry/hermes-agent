"""search_files on a single file survives rg exiting before its group is looked up (#123930).

macOS answers ``getpgid`` with ESRCH for a child that is already tearing down, even
while ``poll()`` still reports it running. A content search of one small file is the
fastest rg run there is, so rg can be exiting before the native lane looks up its
process group. When the lane read the group back with ``getpgid`` instead of recording
rg's own pid (``start_new_session`` makes them equal), the cleanup had no group to fall
back on and ``[Errno 3] No such process`` replaced the matches rg had already produced.

This drives the real handler on the real native rg lane, no mocks. The searches run
from several threads beside threads that hold the GIL, as in a busy desktop process: a
search thread is then often descheduled right after ``Popen`` returns, and rg gets to
exit before the lane runs again. The race only exists on a real macOS kernel, so the
test only runs there.
"""

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

pytestmark = pytest.mark.platforms("macos")

_SEARCH_THREADS = 6
_SPIN_THREADS = 2
_CALLS = 600


def _hold_the_gil(stop: threading.Event) -> None:
    while not stop.is_set():
        pass


def test_single_file_content_search_never_loses_its_matches_to_esrch(tmp_path):
    from tools.file_tools import _get_file_ops, _handle_search_files
    from tools.terminal_tool_lifecycle import cleanup_vm

    target = tmp_path / "haystack.py"
    target.write_text("".join(f"needle_{i:04d} = {i}\n" for i in range(_CALLS)))
    task_id = "t-native-rg-exited-leader"
    ops = _get_file_ops(task_id)
    stop = threading.Event()
    spinners = [threading.Thread(target=_hold_the_gil, args=(stop,), daemon=True) for _ in range(_SPIN_THREADS)]
    old_interval = sys.getswitchinterval()

    def search(i: int):
        # A distinct pattern per call: the repeated-search guard never engages.
        raw = _handle_search_files({"pattern": f"needle_{i:04d}", "path": str(target),
                                    "output_mode": "content", "limit": 3}, task_id=task_id)
        result = json.loads(raw)
        if result.get("error") or not result.get("matches"):
            return i, result.get("error") or raw[:200]
        return None

    try:
        assert ops._native_read_enabled() and ops._has_command("rg"), "native rg lane not active"
        sys.setswitchinterval(0.02)
        for spinner in spinners:
            spinner.start()
        with ThreadPoolExecutor(_SEARCH_THREADS) as pool:
            failures = [f for f in pool.map(search, range(_CALLS)) if f]
        assert not failures, f"{len(failures)}/{_CALLS} single-file searches failed: {failures[:5]}"
    finally:
        stop.set()
        sys.setswitchinterval(old_interval)
        for spinner in spinners:
            if spinner.is_alive():
                spinner.join()
        cleanup_vm(task_id)
