import pickle
import lmdb


class CelebDatabase:
    def __init__(self, db_path):
        self.env = lmdb.open(db_path, max_dbs=1, map_size=10 * 1024 * 1024 * 1024)

    async def write_entry(self, key, data):
        with self.env.begin(write=True) as text:
            text.put(str(key).encode(), pickle.dumps(data))

    async def read_entry(self, key):
        with self.env.begin() as text:
            data = text.get(str(key).encode())
            return pickle.loads(data) if data else None

    async def close(self):
        self.env.close()