with open('.env') as f:
    TELEGRAM_API_TOKEN = f.readline().strip().split('=')[1]
    LOG_GROUP_ID = f.readline().strip().split('=')[1]
LMDB_PATH_FEMALE = 'data/celeb_db_female'
LMDB_PATH_MALE = 'data/celeb_db_male'
FAISS_PATH_FEMALE = 'data/faiss_index_female.bin'
FAISS_PATH_MALE = 'data/faiss_index_male.bin'

NNDB_LMDB_PATH_FEMALE = 'data2/celeb_db_female'
NNDB_LMDB_PATH_MALE = 'data2/celeb_db_male'
NNDB_FAISS_PATH_FEMALE = 'data2/faiss_index_female.bin'
NNDB_FAISS_PATH_MALE = 'data2/faiss_index_male.bin'

DATASET_PATH = 'data'
