import asyncio
from aiogram import Bot, Dispatcher, F, types
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.filters import Command, CommandStart
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, BufferedInputFile
from aiogram.types.message import ContentType
from aiogram.client.default import DefaultBotProperties
from aiogram.enums.parse_mode import ParseMode

import io
import logging
import faiss
from PIL import Image

from config import (
    TELEGRAM_API_TOKEN,
    LMDB_PATH_MALE,
    LMDB_PATH_FEMALE,
    FAISS_PATH_MALE,
    FAISS_PATH_FEMALE,
)
from utils.database import CelebDatabase

from utils.face_embedding import preprocess, get_face_embedding
from utils.search import find_closest


logging.basicConfig(level=logging.INFO)
bot = Bot(
    token=TELEGRAM_API_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML)
)
dp = Dispatcher()

# loaded databases and indexes:
class _Loaded:
    male_lmdb_db = None
    female_lmdb_db = None
    male_faiss_index = None
    female_faiss_index = None

Loaded = _Loaded()

genders = ["Мужской", "Женский"]


def topk_text(gender_id, k, photo_id):
    text = f"""Привет! Этот бот покажет тебе, на какую знаменитость ты похож.

Выберите пол, сколько похожих фотографий выводить

<b>Пол: {genders[gender_id]}</b>

<b>Количество похожих: {k}</b>

<b>{'Фото загружено' if photo_id else 'Теперь загрузите фото'}</b>"""
    return text


def tick(i, j):
    return " ✅" if i == j else ""


def topk_markup(gender_id, k, photo_id):
    inline_keyboard = [
        [
            types.InlineKeyboardButton(
                text=genders[i] + tick(i, gender_id), callback_data=f"g{i}"
            )
            for i in range(len(genders))
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
    photo_id = data.get("photo_id")

    reply = message.reply_to_message
    if photo_id is None and reply is not None and reply.photo is not None:
        photo_id = reply.photo[-1].file_id

    await message.answer(
        topk_text(gender_id, k, photo_id),
        message_effect_id=(
            "5046509860389126442" if message.chat.type == "private" else None
        ),
        reply_markup=topk_markup(gender_id, k, photo_id),
    )
    await state.set_data({"gender": gender_id, "k": k, "photo_id": photo_id})


@dp.callback_query(F.data.startswith("g"))
async def select_gender(call: types.CallbackQuery, state: FSMContext):
    gender_id = int(call.data[1:])

    data = await state.get_data()
    old_gender_id = data.get("gender")
    k = data.get("k")
    photo_id = data.get("photo_id")
    if gender_id == old_gender_id:
        await call.answer("Пол не изменился")
        return

    await call.answer(f"Пол выставлен в {genders[gender_id]}")
    await call.message.edit_text(
        topk_text(gender_id, k, photo_id),
        reply_markup=topk_markup(gender_id, k, photo_id),
    )
    await state.update_data({"gender": gender_id})


@dp.callback_query(F.data.startswith("k"))
async def select_k(call: types.CallbackQuery, state: FSMContext):
    k = int(call.data[1:])

    data = await state.get_data()
    gender_id = data.get("gender")
    old_k = data.get("k")
    photo_id = data.get("photo_id")
    if old_k == k:
        await call.answer("Количество похожих не изменилось")
        return

    await call.answer(f"Количество похожих: {k}")
    await call.message.edit_text(
        topk_text(gender_id, k, photo_id),
        reply_markup=topk_markup(gender_id, k, photo_id),
    )
    await state.update_data({"k": k})


@dp.callback_query(F.data == "launch")
async def inline_launch(call: types.CallbackQuery, state: FSMContext):
    data = await state.get_data()
    await launch(
        message=call.message,
        gender_id=data["gender"],
        k=data["k"],
        photo_id=data["photo_id"],
    )


@dp.message(Command("help"))
async def help_command(message: types.Message):
    # if not message.from_user.id in user_data:
    #     return
    await message.answer(awailable_commands_text)


@dp.message(F.content_type == ContentType.PHOTO)
async def handle_photo(message: types.Message, state: FSMContext):
    data = await state.get_data()
    gender_id = data.get("gender")
    k = data.get("k")
    photo_id = message.photo[-1].file_id
    await state.update_data({"photo_id": photo_id})

    if k is None or gender_id is None:
        await start_command(message, state)
    else:
        await launch(message, gender_id, k, photo_id)
        await state.update_data({"photo_id": None})


async def launch(message: types.Message, gender_id, k, photo_id):
    try:
        if Loaded.male_lmdb_db is None:
            logging.info("Loading MALE LMDB database...")
            Loaded.male_lmdb_db = CelebDatabase(LMDB_PATH_MALE)

            logging.info("Loading FEMALE LMDB database...")
            Loaded.female_lmdb_db = CelebDatabase(LMDB_PATH_FEMALE)

            logging.info("Loading MALE FAISS index...")
            Loaded.male_faiss_index = faiss.read_index(FAISS_PATH_MALE)

            logging.info("Loading FEMALE FAISS index...")
            Loaded.female_faiss_index = faiss.read_index(FAISS_PATH_FEMALE)

        logging.info(f"Received a photo from the user. photo_id: {photo_id}")

        photo_data = await bot.download(photo_id)
        face = await preprocess(photo_data)
        embedding = await get_face_embedding(face)
        results, distances = await find_closest(
            embedding,
            (
                Loaded.male_lmdb_db
                if gender_id == 0
                else Loaded.female_lmdb_db
            ),
            (
                Loaded.male_faiss_index
                if gender_id == 0
                else Loaded.female_faiss_index
            ),
            k
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
