import asyncio
from aiogram import Bot, Dispatcher, F, types
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.filters import Command, CommandStart
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, BufferedInputFile
from aiogram.types.message import ContentType

import io
import logging
import faiss
from PIL import Image

from config import TELEGRAM_API_TOKEN, LMDB_PATH_MALE, LMDB_PATH_FEMALE, FAISS_PATH_MALE, FAISS_PATH_FEMALE
from utils.database import CelebDatabase
from utils.face_embedding import preprocess, get_face_embedding
from utils.search import find_closest


logging.basicConfig(level=logging.INFO)
bot = Bot(token=TELEGRAM_API_TOKEN)
dp = Dispatcher()
user_data = {}


@dp.message(CommandStart())
async def start_command(message: types.Message):
    await message.answer("Привет", message_effect_id='5046509860389126442' if message.chat.type == 'private' else None)


@dp.message(Command("help"))
async def help_command(message: types.Message):
    # YOUR_CODE_HERE
    pass


class TopKState(StatesGroup):
    set_gender = State()
    set_k = State()
    upload_photo = State()


@dp.message(Command("gender"))
async def gender_command(message: types.Message, state: FSMContext):
    gender_keyboard = ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="Мужской"), KeyboardButton(text="Женский")]],
        resize_keyboard=True,
        one_time_keyboard=True,
    )
    await message.answer("Пожалуйста, выберете ваш пол:", reply_markup=gender_keyboard)
    await state.set_state(TopKState.set_gender)


@dp.message(TopKState.set_gender, F.text.in_(["Мужской", "Женский"]))
async def set_gender(message: types.Message, state: FSMContext):
    user_data[message.from_user.id] = {"gender": message.text}
    await message.answer(
        f"Ваш пол был выставлен в <b>{message.text}</b>.", reply_markup=types.ReplyKeyboardRemove()
    )
    await state.clear()


@dp.message(TopKState.set_gender)
async def invalid_gender(message: types.Message):
    # YOUR_CODE_HERE
    pass


@dp.message(Command("topk"))
async def topk_command(message: types.Message, state: FSMContext):
    user_id = message.from_user.id
    if user_id not in user_data or "gender" not in user_data[user_id]:
        await message.answer("Вы должны выбрать свой пол /gender.")
        return

    topk_keyboard = ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text=str(i))] for i in range(1, 6)],
        resize_keyboard=True,
        one_time_keyboard=True,
    )
    await message.answer("Выберете значение k (целое число от 1 до 5):",
                         reply_markup=topk_keyboard)
    await state.set_state(TopKState.set_k)


@dp.message(TopKState.set_k)
async def set_k(message: types.Message, state: FSMContext):
    try:
        k = int(message.text)
        if k <= 0 or k > 5:
            raise ValueError

        await state.update_data(k=k)
        await message.answer("Теперь, пожалуйста, загрузите фото")
        await state.set_state(TopKState.upload_photo)
    except Exception as e:
        logging.error(f"Error processing photo: {e}")
        await message.answer("Некорректный ввод. Нужно целое чило от 1 до 5")


@dp.message(TopKState.upload_photo, F.content_type == ContentType.PHOTO)
async def handle_photo(message: types.Message, state: FSMContext):
    data = await state.get_data()
    k = data.get("k")
    gender = user_data[message.from_user.id]['gender']

    if not k:
        await message.answer("Возникла ошибка. Пожалуйста, перезапустите командy /topk")
        await state.clear()
        return

    try:
        # YOUR_CODE_HERE

        logging.info("Received a photo from the user.")

        # YOUR_CODE_HERE
    except Exception as e:
        logging.error(f"Error processing photo: {e}")
        # YOUR_CODE_HERE

    await state.clear()


@dp.message(TopKState.upload_photo)
async def non_photo_upload(message: types.Message):
    message.answer("Вы отправили не фотографию, отправьте фотографию")


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