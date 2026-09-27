import sqlite3
from typing import List, Set, Iterable, Dict, Tuple

MAX_CANDIDATES_PER_ENTITY = 15

def name_tokens(name: str) -> frozenset:
    if not name: return frozenset()
    parts = name.replace('-', ' ').split()
    return frozenset(p for p in parts if len(p) > 2)

def all_keys(name: str, address: str) -> List[str]:
    keys = set()
    nt = name_tokens(name)
    for t in nt:
        keys.add(f"nt:{t}")
    if address:
        parts = address.replace('-', ' ').replace(',', ' ').split()
        for p in parts:
            if len(p) > 3:
                keys.add(f"at:{p}")
    return list(keys)

class DiskBlockingIndex:
    def __init__(self, db_path: str, fresh=False):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, isolation_level=None)

    def get_candidates_for_keys_batch(self, keys: Set[str]) -> Dict[str, Set[str]]:
        res = {}
        if not keys: return res
        chunk_size = 500
        keys_list = list(keys)
        for i in range(0, len(keys_list), chunk_size):
            chunk = keys_list[i:i+chunk_size]
            ph = ','.join(['?']*len(chunk))
            cur = self.conn.execute(f'SELECT block_key, entity_id FROM block_entries WHERE block_key IN ({ph})', chunk)
            for k, eid in cur:
                if k not in res:
                    res[k] = set()
                res[k].add(str(eid))
        return res
