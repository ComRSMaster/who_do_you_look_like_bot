with open('.env') as f:
    TELEGRAM_API_TOKEN = f.readline().strip().split('=')[1]
    LOG_GROUP_ID = f.readline().strip().split('=')[1]
    REDIS_URL = f.readline().strip().split('=')[1]
LMDB_PATH_FEMALE = 'data/celeb_db_female2'
LMDB_PATH_MALE = 'data/celeb_db_male2'
FAISS_PATH_FEMALE = 'data/faiss_index_female2.bin'
FAISS_PATH_MALE = 'data/faiss_index_male2.bin'

NNDB_LMDB_PATH_FEMALE = 'nndb_data/celeb_db_female'
NNDB_LMDB_PATH_MALE = 'nndb_data/celeb_db_male'
NNDB_FAISS_PATH_FEMALE = 'nndb_data/faiss_index_female.bin'
NNDB_FAISS_PATH_MALE = 'nndb_data/faiss_index_male.bin'

DATASET_PATH = 'data'
