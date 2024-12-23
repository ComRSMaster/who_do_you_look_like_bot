import os
import numpy as np
import csv

import scipy.io
import pandas as pd
from datetime import datetime, timedelta

import asyncio
from tqdm.asyncio import tqdm

import faiss

from utils.database import CelebDatabase
from config import LMDB_PATH_MALE, LMDB_PATH_FEMALE, FAISS_PATH_MALE, FAISS_PATH_FEMALE, DATASET_PATH
from config import NNDB_PATH_MALE, NNDB_PATH_FEMALE, NNDB_FAISS_PATH_MALE, NNDB_FAISS_PATH_FEMALE, NNDB_DATASET_PATH
from utils.face_embedding import preprocess, get_face_embedding


async def get_best_images(): # получаем лучшие фотографии каждого человека
    mat_data = scipy.io.loadmat(os.path.join(DATASET_PATH, 'imdb_crop/imdb.mat')) # загрузка базы данных
    dt = mat_data['imdb'][0, 0]  # извлекаем данные:
    keys_s = ('gender', 'dob', 'photo_taken',
              'face_score', 'second_face_score', 'celeb_id')
    values = {k: dt[k].squeeze() for k in keys_s}
    keys_n = ('full_path', 'name') # создаем пути к изображениям:
    for k in keys_n:
        values[k] = np.array([x if not x else x[0] for x in dt[k][0]])
    # обработка местоположения лица
    values['face_location'] =\
        [tuple(x[0].tolist()) for x in dt['face_location'].squeeze()]
    
    set_nrows = {len(v) for _, v in values.items()} # убедимся, что все массивы имеют одну длину:
    assert len(set_nrows) == 1 

    df_values = pd.DataFrame(values) 
    matlab_origin = datetime(1, 1, 1)
    days_offset = timedelta(days=366)

    def matlab_datenum_to_datetime(datenum):
        try:
            if datenum > 0 and datenum < 3652059:
                return matlab_origin + timedelta(days=datenum) - days_offset
            else:
                return pd.NaT
        except OverflowError:
            return pd.NaT

    df_values['dob'] = df_values['dob'].apply(matlab_datenum_to_datetime) # преобразуем даты в формат pandas, отбираем подходящие фото:
    filtered_df = df_values[(df_values['face_score'] > 0) & (df_values['second_face_score'].isna())]

    best_images = ( # сортировка по счету лица и даты фото:
        filtered_df.sort_values(by=['face_score', 'photo_taken'], ascending=[False, False])
        .groupby('celeb_id')
        .first()
        .reset_index()
    )
    best_images = best_images.drop(columns=['second_face_score', 'celeb_id'])

    return (
        best_images[best_images['gender'] == 0]['full_path'].values,
        best_images[best_images['gender'] == 0]['name'].values,
        best_images[best_images['gender'] == 1]['full_path'].values,
        best_images[best_images['gender'] == 1]['name'].values
    )

async def get_nndb_images():
    big_path = NNDB_DATASET_PATH + "/data_with_paths.csv"
    female_paths = []
    female_names = []
    male_paths = []
    male_names = []
    with open(big_path, encoding='utf-8') as r_file:
        file_reader = csv.reader(r_file, delimiter = ";")
        count = 0
        for row in file_reader:
            if(count > 0):
                name = row[0]
                gender = row[1]
                url = row[2]
                path = NNDB_DATASET_PATH + "/" + row[3]
                if(gender == "Female"):
                    female_paths.append(path)
                    female_names.append(name)
                else:
                    male_paths.append(path)
                    male_names.append(name)
            count += 1
    return (
        female_paths, female_names, male_paths, male_names
    )

            



async def process_image(key, path, name, lmdb_db, all_embeddings): # делает ембеддинг модели по ключу, пути, имени и датабазы
    if not path.lower().endswith(('.png', '.jpg', '.jpeg')):
        return

    image_path = os.path.join(DATASET_PATH, 'imdb_crop', path) # получаем путь до фото

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


async def faiss_build(model, PATH_FEMALE, FAISS_FEMALE, PATH_MALE, FAISS_MALE):
    if not os.path.exists(PATH_FEMALE): # создает каталог, если его нет
        os.makedirs(PATH_FEMALE)
    female_lmdb_db = CelebDatabase(PATH_FEMALE)
    if not os.path.exists(PATH_MALE): # создает каталог, если его нет
        os.makedirs(PATH_MALE)
    male_lmdb_db = CelebDatabase(PATH_MALE)

    print("\nGetting best images...")
    if(model == "lmdb"):
        female_paths, female_names, male_paths, male_names = await get_best_images() # получает лучшие изображения
    else:
        female_paths, female_names, male_paths, male_names = await get_nndb_images() # получает лучшие изображения
    print("\nGot best images...")

    female_embeddings = []
    male_embeddings = []
    tasks = [
        process_image(key, female_paths[key], female_names[key], female_lmdb_db, female_embeddings)
        for key in range(len(female_names))
    ]
    tasks += [
        process_image(key, male_paths[key], male_names[key], male_lmdb_db, male_embeddings)
        for key in range(len(male_names))
    ]
    await tqdm.gather(*tasks) # "параллельное" выполнение корутин для более эффективной работы
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
    index_build(female_embeddings, FAISS_FEMALE) # строим faiss для женщин
    print(f"FAISS index for female saved to '{FAISS_FEMALE}'")

    print("\nCreating FAISS index for male...")
    index_build(male_embeddings, FAISS_MALE) # строим faiss для мужщин
    print(f"FAISS index for male saved to '{FAISS_MALE}'")

async def build():
    print("\nBuild has been started")
    if not os.path.exists(DATASET_PATH): # проверка есть ли файл DATASET_PATH
        raise FileNotFoundError(f"Celebrity dataset directory '{DATASET_PATH}' not found.")
    if not os.path.exists(NNDB_DATASET_PATH): # проверка есть ли файл NNDB_DATASET_PATH
        raise FileNotFoundError(f"Celebrity dataset directory '{NNDB_DATASET_PATH}' not found.")
    
    await faiss_build("nndb", NNDB_PATH_FEMALE, NNDB_FAISS_PATH_FEMALE, NNDB_PATH_MALE, NNDB_FAISS_PATH_MALE)
    await faiss_build("lmdb", LMDB_PATH_FEMALE, FAISS_PATH_FEMALE, LMDB_PATH_MALE, FAISS_PATH_MALE)
    
    


if __name__ == "__main__":
    asyncio.run(build())