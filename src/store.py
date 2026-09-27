import sqlite3
from typing import Dict, Set, Tuple

class RecordStore:
    def __init__(self, db_path: str, fresh=False):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, isolation_level=None)

    def get_records_batch(self, entity_ids: Set[str]) -> Dict[str, Tuple[str, str, str]]:
        if not entity_ids: return {}
        res = {}
        chunk_size = 500
        eids_list = list(entity_ids)
        for i in range(0, len(eids_list), chunk_size):
            chunk = eids_list[i:i+chunk_size]
            ph = ','.join(['?']*len(chunk))
            cur = self.conn.execute(f'SELECT entity_id, name, address, country FROM records WHERE entity_id IN ({ph})', chunk)
            for row in cur:
                res[str(row[0])] = (str(row[1] or ''), str(row[2] or ''), str(row[3] or ''))
        return res
