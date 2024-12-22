import numpy as np


async def find_closest(embedding, lmdb_database, faiss_index, k=1):
     distances, result_idx = faiss_index.search(np.array([embedding]), k) # поиск ближайших k фотографий
     distances = distances[0].tolist()  # косинусное расстояние до найденных фото
     closest_entries = [] # имена похожих знаменитостей:
     for key in result_idx[0]:
        closest_entries.append(await lmdb_database.read_entry(key))
     return closest_entries, distances