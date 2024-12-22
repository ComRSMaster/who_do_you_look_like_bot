import os

with open('.env') as f:
    TELEGRAM_API_TOKEN = f.readline().split('=')[1]
LMDB_PATH_FEMALE = 'data/celeb_db_female'
LMDB_PATH_MALE = 'data/celeb_db_male'
FAISS_PATH_FEMALE = 'data/faiss_index_female.bin'
FAISS_PATH_MALE = 'data/faiss_index_male.bin'
DATASET_PATH = 'data'
