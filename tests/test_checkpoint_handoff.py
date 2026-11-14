"""CPU-only recovery checks; never inspect or signal production process IDs."""
import errno
import fcntl
import importlib.util
import json
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import de_checkpoint_handoff as H


class HandoffSafety(unittest.TestCase):
    saved = dict(pid=91000001, ppid=91000000, startticks=123, argv=['fixture'], state='T')

    def test_disappearance_is_absence_but_permission_error_is_not(self):
        for code in (errno.ENOENT, errno.ESRCH):
            with mock.patch.object(H, 'raw_identity', side_effect=OSError(code, 'fixture')):
                self.assertIsNone(H.identity(self.saved['pid']))
        with mock.patch.object(H, 'raw_identity', side_effect=PermissionError(errno.EACCES, 'fixture')):
            with self.assertRaises(PermissionError): H.identity(self.saved['pid'])

    def test_exit_between_identity_and_signal_is_safe(self):
        with mock.patch.object(H, 'identity', return_value=self.saved), mock.patch.object(H.os, 'kill', side_effect=ProcessLookupError):
            self.assertFalse(H.signal_saved(self.saved, signal.SIGCONT))

    def test_mismatched_process_is_never_signalled(self):
        for key, value in [('startticks', 124), ('argv', ['unrelated']), ('ppid', 999)]:
            with mock.patch.object(H, 'identity', return_value={**self.saved, key: value}), mock.patch.object(H.os, 'kill') as kill:
                with self.assertRaises(AssertionError): H.signal_saved(self.saved, signal.SIGTERM)
                kill.assert_not_called()

    def test_reparented_child_requires_absent_original_parent(self):
        orphan = {**self.saved, 'ppid': 999}
        with mock.patch.object(H, 'identity', side_effect=[orphan, self.saved]), mock.patch.object(H.os, 'kill') as kill:
            with self.assertRaises(AssertionError): H.signal_saved(self.saved, signal.SIGTERM, True)
            kill.assert_not_called()
        with mock.patch.object(H, 'identity', side_effect=[orphan, None]), mock.patch.object(H.os, 'kill') as kill:
            self.assertTrue(H.signal_saved(self.saved, signal.SIGTERM, True))
            kill.assert_called_once_with(self.saved['pid'], signal.SIGTERM)
        with mock.patch.object(H, 'identity', side_effect=[orphan, {**self.saved, 'state': 'Z'}]), mock.patch.object(H.os, 'kill'):
            self.assertTrue(H.signal_saved(self.saved, signal.SIGTERM, True))

    def test_restore_child_before_parent(self):
        child = {**self.saved, 'pid': 91000002}
        with mock.patch.object(H, 'identity', return_value=self.saved), mock.patch.object(H, 'signal_saved') as send:
            H.restore_paused({'jobs': [{'child': child, 'parent': self.saved}]})
            self.assertEqual(send.call_args_list, [mock.call(child, signal.SIGCONT), mock.call(self.saved, signal.SIGCONT)])

    def test_atomic_snapshot_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); src = base / 'src'; dst = base / 'dst'
            src.write_bytes(b'checkpoint fixture' * 1000)
            record = H.atomic_copy(src, dst)
            self.assertEqual(record['sha256'], H.Q.sha(src))
            self.assertEqual(dst.read_bytes(), src.read_bytes())
            with self.assertRaises(AssertionError): H.atomic_copy(src, dst)
            self.assertFalse((base / 'dst.tmp').exists())

    def test_abandoned_temporary_does_not_prevent_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); src = base / 'src'; dst = base / 'dst'
            src.write_bytes(b'complete'); stale = base / 'dst.tmp-abandoned'; stale.write_bytes(b'partial')
            H.atomic_copy(src, dst)
            self.assertEqual(dst.read_bytes(), b'complete')
            self.assertEqual(stale.read_bytes(), b'partial')

    def test_native_tombstone_blocks_sharded_admission_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); journal = base / 'journal.json'
            first = H.close_sharded_admission(base, 3, journal)
            self.assertEqual(first['status'], 'native_recovery_committed')
            self.assertEqual(H.close_sharded_admission(base, 3, journal), first)
            self.assertTrue((base / 'sharded-search-s3.json').exists())
            with self.assertRaises(AssertionError): H.close_sharded_admission(base, 3, base / 'different.json')

    def test_native_fallback_refuses_live_sharded_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            with (base / 'sharded-search-s3.lock').open('a') as admitted:
                fcntl.flock(admitted, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError): H.close_sharded_admission(base, 3, base / 'journal.json')
            self.assertFalse((base / 'sharded-search-s3.json').exists())

    def test_native_fallback_preserves_existing_execution_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); path = base / 'sharded-search-s3.json'
            path.write_text(json.dumps({'status': 'failed', 'seed': 3}))
            before = path.read_bytes()
            with self.assertRaises(AssertionError): H.close_sharded_admission(base, 3, base / 'journal.json')
            self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__': unittest.main()
