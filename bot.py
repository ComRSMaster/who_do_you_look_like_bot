import asyncio
from aiogram import Bot, Dispatcher, F, types
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.redis import RedisStorage
from aiogram.fsm.strategy import FSMStrategy
from aiogram.filters import Command, CommandStart
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, BufferedInputFile
from aiogram.types.message import ContentType
from aiogram.client.default import DefaultBotProperties
from aiogram.enums.parse_mode import ParseMode
from aiogram.enums.chat_type import ChatType
from aiogram.enums.update_type import UpdateType

import io
import logging
import traceback
import faiss
import html
from PIL import Image

from config import *
from utils.database import CelebDatabase

from utils.face_embedding import preprocess, get_face_embedding
from utils.search import find_closest


logging.basicConfig(level=logging.INFO)
bot = Bot(
    token=TELEGRAM_API_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML)
)
dp = Dispatcher(storage=RedisStorage.from_url(REDIS_URL), fsm_strategy=FSMStrategy.CHAT)


# loaded databases and indexes:
class _Loaded:
    male_lmdb_db = None
    male_faiss_index = None

    female_lmdb_db = None
    female_faiss_index = None

    nndb_male_lmdb_db = None
    nndb_male_faiss_index = None

    nndb_female_lmdb_db = None
    nndb_female_faiss_index = None


Loaded = _Loaded()

genders = ["Мужской", "Женский"]
models = ["IMDB_WIKI", "NNDB_CELEBS"]


def topk_text(gender_id, k, model_id, photo_id):
    text = f"""Привет! Этот бот покажет тебе, на какую знаменитость ты похож.

Выберите пол, сколько похожих фотографий выводить

<b>Пол: {genders[gender_id]}</b>

<b>База данных: {models[model_id]}</b>

<b>Количество похожих: {k}</b>

<b>{'Фото загружено' if photo_id else 'Теперь загрузите фото'}</b>"""
    return text


def tick(i, j):
    return " ✅" if i == j else ""


def topk_markup(gender_id, k, model_id, photo_id):
    inline_keyboard = [
        [
            types.InlineKeyboardButton(
                text=genders[i] + tick(i, gender_id), callback_data=f"g{i}"
            )
            for i in range(len(genders))
        ],
        [
            types.InlineKeyboardButton(
                text=models[i] + tick(i, model_id), callback_data=f"m{i}"
            )
            for i in range(len(models))
        ],
        [
            types.InlineKeyboardButton(text=str(i) + tick(i, k), callback_data=f"k{i}")
            for i in range(1, 6)
        ],
    ]
    if photo_id:
        inline_keyboard.append(
            [
                types.InlineKeyboardButton(
                    text="🚀 Продолжить 🚀", callback_data="launch"
                )
            ]
        )
    return types.InlineKeyboardMarkup(inline_keyboard=inline_keyboard)


@dp.message(CommandStart())
async def start_command(message: types.Message, state: FSMContext):
    # message_effect_id: https://stackoverflow.com/questions/78600012/message-effect-id-in-telegram-bot-api
    data = await state.get_data()
    gender_id = data.get("gender", 0)
    k = data.get("k", 5)
    model_id = data.get("model_id", 1)
    photo_id = data.get("photo_id")

    reply = message.reply_to_message
    if photo_id is None and reply is not None and reply.photo is not None:
        photo_id = reply.photo[-1].file_id

    await message.answer(
        topk_text(gender_id, k, model_id, photo_id),
        message_effect_id=(
            "5046509860389126442" if message.chat.type == ChatType.PRIVATE else None
        ),
        reply_markup=topk_markup(gender_id, k, model_id, photo_id),
    )
    await state.set_data({"gender": gender_id, "k": k, "model_id": model_id, "photo_id": photo_id})


@dp.callback_query(F.data.startswith("g"))
async def select_gender(call: types.CallbackQuery, state: FSMContext):
    gender_id = int(call.data[1:])

    data = await state.get_data()
    old_gender_id = data.get("gender", 0)
    k = data.get("k", 5)
    model_id = data.get("model_id", 1)
    photo_id = data.get("photo_id")
    if gender_id == old_gender_id:
        await call.answer("Пол не изменился")
        return

    await call.answer(f"Пол выставлен в {genders[gender_id]}")
    await call.message.edit_text(
        topk_text(gender_id, k, model_id, photo_id),
        reply_markup=topk_markup(gender_id, k, model_id, photo_id),
    )
    await state.update_data({"gender": gender_id})


@dp.callback_query(F.data.startswith("k"))
async def select_k(call: types.CallbackQuery, state: FSMContext):
    k = int(call.data[1:])

    data = await state.get_data()
    gender_id = data.get("gender", 0)
    old_k = data.get("k", 5)
    model_id = data.get("model_id", 1)
    photo_id = data.get("photo_id")
    if old_k == k:
        await call.answer("Количество похожих не изменилось")
        return

    await call.answer(f"Количество похожих: {k}")
    await call.message.edit_text(
        topk_text(gender_id, k, model_id, photo_id),
        reply_markup=topk_markup(gender_id, k, model_id, photo_id),
    )
    await state.update_data({"k": k})


@dp.callback_query(F.data.startswith("m"))
async def select_model(call: types.CallbackQuery, state: FSMContext):
    model_id = int(call.data[1:])

    data = await state.get_data()    
    gender_id = data.get("gender", 0)
    k = data.get("k", 5)
    old_model_id = data.get("model_id", 1)
    photo_id = data.get("photo_id")
    if old_model_id == model_id:
        await call.answer("Модель не изменилась")
        return

    await call.answer(f"Выбрана модель: {models[model_id]}")
    await call.message.edit_text(
        topk_text(gender_id, k, model_id, photo_id),
        reply_markup=topk_markup(gender_id, k, model_id, photo_id),
    )
    await state.update_data({"model_id": model_id})



@dp.callback_query(F.data == "launch")
async def inline_launch(call: types.CallbackQuery, state: FSMContext):
    data = await state.get_data()
    await launch(
        message=call.message,
        gender_id=data["gender"],
        k=data["k"],
        model_id=data["model_id"],
        photo_id=data["photo_id"],
    )


# @dp.message(Command("help"))
# async def help_command(message: types.Message):
#     # if not message.from_user.id in user_data:
#     #     return
#     await message.answer(awailable_commands_text)


@dp.message(F.content_type == ContentType.PHOTO)
async def handle_photo(message: types.Message, state: FSMContext):
    data = await state.get_data()
    gender_id = data.get("gender")
    k = data.get("k")
    model_id = data.get("model_id")
    photo_id = message.photo[-1].file_id
    await state.update_data({"photo_id": photo_id})

    if k is None or gender_id is None or model_id is None:
        await start_command(message, state)
    else:
        await launch(message, gender_id, k, model_id, photo_id)
        # await state.update_data({"photo_id": None})


async def launch(message: types.Message, gender_id, k, model_id, photo_id):
    try:
        if model_id == 0:
            if gender_id == 0 and Loaded.male_lmdb_db is None:
                logging.info("Loading MALE LMDB database...")
                Loaded.male_lmdb_db = CelebDatabase(LMDB_PATH_MALE)

                logging.info("Loading MALE FAISS index...")
                Loaded.male_faiss_index = faiss.read_index(FAISS_PATH_MALE)
            elif gender_id == 1 and Loaded.female_lmdb_db is None:
                logging.info("Loading FEMALE LMDB database...")
                Loaded.female_lmdb_db = CelebDatabase(LMDB_PATH_FEMALE)

                logging.info("Loading FEMALE FAISS index...")
                Loaded.female_faiss_index = faiss.read_index(FAISS_PATH_FEMALE)
        elif model_id == 1:
            # NNDB
            if gender_id == 0 and Loaded.nndb_male_lmdb_db is None:
                logging.info("Loading NNDB MALE LMDB database...")
                Loaded.nndb_male_lmdb_db = CelebDatabase(NNDB_LMDB_PATH_MALE)

                logging.info("Loading NNDB MALE FAISS index...")
                Loaded.nndb_male_faiss_index = faiss.read_index(NNDB_FAISS_PATH_MALE)
            elif gender_id == 1 and Loaded.nndb_female_lmdb_db is None:
                logging.info("Loading NNDB FEMALE LMDB database...")
                Loaded.nndb_female_lmdb_db = CelebDatabase(NNDB_LMDB_PATH_FEMALE)

                logging.info("Loading NNDB FEMALE FAISS index...")
                Loaded.nndb_female_faiss_index = faiss.read_index(NNDB_FAISS_PATH_FEMALE)

        logging.info(f"Received a photo from the user. photo_id: {photo_id}")

        photo_data = await bot.download(photo_id)
        # await message.answer_photo(photo_id, "Ваше фото:")
        face = await preprocess(photo_data)
        embedding = await get_face_embedding(face)
        if model_id == 0:
            lmdb_database = Loaded.male_lmdb_db if gender_id == 0 else Loaded.female_lmdb_db
            faiss_index = Loaded.male_faiss_index if gender_id == 0 else Loaded.female_faiss_index
        else:
            lmdb_database = Loaded.nndb_male_lmdb_db if gender_id == 0 else Loaded.nndb_female_lmdb_db
            faiss_index = Loaded.nndb_male_faiss_index if gender_id == 0 else Loaded.nndb_female_faiss_index
        results, distances = await find_closest(
            embedding,
            lmdb_database,
            faiss_index,
            k,
        )

        if results:
            for result, distance in zip(results, distances):
                name = result["name"]
                image = Image.open(io.BytesIO(result["photo"]))

                with io.BytesIO() as buffer:
                    image.save(buffer, format="JPEG")
                    buffer.seek(0)
                    file_to_send = BufferedInputFile(buffer.getvalue(), "photo.jpg")

                    await message.answer_photo(
                        photo=file_to_send,
                        caption=f'Схожесть с <a href="https://ya.ru/search/?text={name}">{name}</a> на {int(round(distance, 2) * 100)}%',
                    )
        else:
            await message.answer(
                "Не получилось найти похожее лицо. Попробуйте другое фото."
            )
    except Exception as e:
        logging.error(f"Error processing photo: {e}", exc_info=True)
        await message.answer(
            "Произошла ошибка при обработке фотографии. Возможно на ней нет лица. Попробуйте ещё раз."
        )


@dp.message()
async def unknown_command(message: types.Message):
    message.answer("Неизвестная команда")


current_chat = None
current_user = None


async def log_message(message: types.Message):
    global current_chat, current_user

    chat = message.chat
    user = message.from_user
    if current_chat != chat.id or current_user != user.id:
        current_chat = chat.id
        current_user = user.id
        user_text = f"<b>Пользователь:</b>\nID: {user.id}\n{user.mention_html('Имя: ' + user.full_name)}{'' if user.username is None else '\n@' + user.username}"

        if chat.type != ChatType.PRIVATE:
            user_text = (
                f"<b>Группа:</b>\nID: {chat.id}\nИмя: {chat.full_name}{'' if chat.username is None else '\n@' + chat.username}{'' if chat.invite_link is None else '\n' + chat.invite_link}\n\n"
                + user_text
            )
        await bot.send_message(LOG_GROUP_ID, user_text, disable_notification=True)
    await message.copy_to(LOG_GROUP_ID, disable_notification=True)


@dp.update.middleware()
async def log_middleware(handler, event: types.Update, data):
    try:
        if event.message is not None:
            asyncio.create_task(log_message(event.message))
        return await handler(event, data)
    except Exception as e:
        logging.error(e, exc_info=True)
        exc_text = "<b>ОШИБКА:</b>\n" + html.escape(
            "".join(traceback.format_exception(e))
        )
        await bot.send_message(LOG_GROUP_ID, exc_text[:4096], disable_notification=True)


async def main():
    try:
        logging.info("Starting bot...")
        await dp.start_polling(bot)
    except Exception as e:
        logging.error(f"Error processing photo: {e}")
    finally:
        # YOUR_CODE_HERE
        pass


if __name__ == "__main__":
    asyncio.run(main())
