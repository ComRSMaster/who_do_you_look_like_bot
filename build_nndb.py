import os
import numpy as np

import scipy.io
import pandas as pd
from datetime import datetime, timedelta

import asyncio
from tqdm.asyncio import tqdm

import faiss

from utils.database import CelebDatabase
from config import LMDB_PATH_MALE, LMDB_PATH_FEMALE, FAISS_PATH_MALE, FAISS_PATH_FEMALE, DATASET_PATH
from utils.face_embedding import preprocess, get_face_embedding


async def get_best_images(): # получаем лучшие фотографии каждого человека
    with
    return (
        best_images[best_images['gender'] == 0]['full_path'].values,
        best_images[best_images['gender'] == 0]['name'].values,
        best_images[best_images['gender'] == 1]['full_path'].values,
        best_images[best_images['gender'] == 1]['name'].values
    )


async def process_image(key, path, name, lmdb_db, all_embeddings): # делает ембеддинг модели по ключу, пути, имени и датабазы
    if not path.lower().endswith(('.png', '.jpg', '.jpeg')):
        return

    image_path = os.path.join(DATASET_PATH, 'nndb_celeb', path) # получаем путь до фото

    try:
        face = await preprocess(image_path)
        embedding = await get_face_embedding(face) # получили эмбеддинг фото

        with open(image_path, 'rb') as f:
            photo_bytes = f.read() # прочитали фото в бинарном формате

        data = { # словарь с нужными параметрами: имя, ембеддинги и фото
            'name': name,
            'embedding': embedding.tolist(),
            'photo': photo_bytes,
        }
        await lmdb_db.write_entry(key, data) # записываем в датабазу 
        all_embeddings.append((embedding, key)) # записываем в ембеддинги
    except Exception as e:
        all_embeddings.append((np.zeros(512), key)) # very important!
        print(f"Error processing {path}: {e}")


async def build():
    print("\nBuild has been started")
    if not os.path.exists(DATASET_PATH): # проверка есть ли файл DATASET_PATH
        raise FileNotFoundError(f"Celebrity dataset directory '{DATASET_PATH}' not found.")
    
    if not os.path.exists(LMDB_PATH_FEMALE): # создает каталог, если его нет
        os.makedirs(LMDB_PATH_FEMALE)
    female_lmdb_db = CelebDatabase(LMDB_PATH_FEMALE)
    if not os.path.exists(LMDB_PATH_MALE): # создает каталог, если его нет
        os.makedirs(LMDB_PATH_MALE)
    male_lmdb_db = CelebDatabase(LMDB_PATH_MALE)

    print("\nGetting best images...")
    female_paths, female_names, male_paths, male_names = await get_best_images() # получает лучшие изображения
    print("\nGot best images...")

    female_embeddings = []
    male_embeddings = []
    
    for key in range(len(female_names)):
        await process_image(key, female_paths[key], female_names[key], female_lmdb_db, female_embeddings)
    
    for key in range(len(male_names)):
        await process_image(key, male_paths[key], male_names[key], male_lmdb_db, male_embeddings)
    # tasks = [
    #     process_image(key, female_paths[key], female_names[key], female_lmdb_db, female_embeddings)
    #     for key in range(len(female_names))
    # ]
    # tasks += [
    #     process_image(key, male_paths[key], male_names[key], male_lmdb_db, male_embeddings)
    #     for key in range(len(male_names))
    # ]
    # await tqdm.gather(*tasks) # "параллельное" выполнение корутин для более эффективной работы
    await female_lmdb_db.close() # закрывает файл
    await male_lmdb_db.close() # закрывает файл

    def index_build(embeddings, FAISS_PATH_GENDER):
        dimension = embeddings[0][0].shape[0]
        index = faiss.IndexFlatIP(dimension) # обучение модели
        embeddings = [emb for emb, _ in sorted(embeddings, key=lambda x: x[1])] # получение эмбеддингов и сортировка по ключам 
        fmap = np.array(embeddings) # numpy array эмбеддингов
        index.add(fmap) # добавили эмбеддинги в index
        faiss.write_index(index, FAISS_PATH_GENDER) # сохранение index по пути FAISS_PATH_GENDER

    print("\nCreating FAISS index for female...")
    index_build(female_embeddings, FAISS_PATH_FEMALE) # строим faiss для женщин
    print(f"FAISS index for female saved to '{FAISS_PATH_FEMALE}'")

    print("\nCreating FAISS index for male...")
    index_build(male_embeddings, FAISS_PATH_MALE) # строим faiss для мужщин
    print(f"FAISS index for male saved to '{FAISS_PATH_MALE}'")


if __name__ == "__main__":
    asyncio.run(build())