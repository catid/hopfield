"""Two-owned-worker barrier preventing host archive/answer overlap."""
import hashlib
import json
import os
from pathlib import Path
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PairBarrier:
    def __init__(self, directory, gpu, protocol_hash, peer_state, peer_failure,
                 watchdog, maximum_wait_seconds=60):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.gpu = gpu
        self.protocol_hash = protocol_hash
        self.peer_state = Path(peer_state)
        self.peer_failure = Path(peer_failure)
        self.watchdog = watchdog
        self.maximum_wait_seconds = maximum_wait_seconds
        self.events = []

    def gate(self, stage):
        if not stage or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in stage):
            raise ValueError('Unsafe barrier stage')
        started = time.perf_counter()
        own = self.directory / f'{stage}-gpu{self.gpu}.json'
        peer = self.directory / f'{stage}-gpu{1 - self.gpu}.json'
        temporary = own.with_name(own.name + f'.tmp-{os.getpid()}')
        payload = {'stage': stage, 'gpu': self.gpu, 'pid': os.getpid(),
                   'protocol_sha256': self.protocol_hash, 'arrived_monotonic': started}
        with temporary.open('x') as stream:
            stream.write(json.dumps(payload) + '\n')
        try:
            # Atomic publication that cannot replace an earlier attempt marker.
            os.link(temporary, own)
        finally:
            temporary.unlink()
        while True:
            self.watchdog()
            if self.peer_failure.exists():
                raise RuntimeError('Peer failure at host-wide answer barrier')
            state = None
            if self.peer_state.exists():
                state = json.loads(self.peer_state.read_text())
                if state['status'] in ['incomplete', 'failed', 'budget_exhausted']:
                    raise RuntimeError('Peer incomplete at host-wide answer barrier')
            if state is not None:
                if state['status'] == 'complete':
                    if stage != 'archives-done' or not peer.exists():
                        raise RuntimeError('Peer completed before this required barrier')
                else:
                    try:
                        os.kill(int(state['pid']), 0)
                    except ProcessLookupError as error:
                        raise RuntimeError('Owned peer process exited before barrier') from error
            if peer.exists():
                other = json.loads(peer.read_text())
                assert other['stage'] == stage and other['gpu'] == 1 - self.gpu
                assert other['protocol_sha256'] == self.protocol_hash
                waited = time.perf_counter() - started
                self.events.append({'stage': stage, 'own_marker': own.name, 'peer_marker': peer.name,
                                    'own_sha256': sha(own), 'peer_sha256': sha(peer),
                                    'wait_seconds': waited})
                return waited
            if state is not None and state['status'] == 'complete':
                raise RuntimeError('Peer completed without reaching this required barrier')
            if time.perf_counter() - started > self.maximum_wait_seconds:
                raise TimeoutError('Host-wide answer barrier wait cap exhausted; no retry')
            time.sleep(.005)
