import os
import telebot
from telebot import apihelper 
from gigachat import GigaChat
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# === НАСТРОЙКИ КЛЮЧЕЙ И ПУТЕЙ ===
TELEGRAM_TOKEN = "8856778026:AAGfA0V9PkB2fhu9STcIW3kNuGmVFQEPs9E"
GIGACHAT_CREDENTIALS = "MDFhMGMyYzItNjk5Yy03MjQ2LWIwMjUtNzliYmE2OTQ5NzQ0OmM0OTEzZDUxLTc1M2QtNDkwYS04NTc2LTRkNGRkZWQyNGI2Yw=="
DB_DIR = "faiss_index"

print("Инициализация компонентов бота...")

# 1. Запуск Telegram-бота
bot = telebot.TeleBot(TELEGRAM_TOKEN)

# 2. Подгружаем модель эмбеддингов (ту же самую, что и при создании базы)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# 3. Подключаем нашу созданную векторную базу данных FAISS
if os.path.exists(DB_DIR):
    # allow_dangerous_deserialization=True нужен для локальной загрузки файлов FAISS на Windows
    db = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
    print("Векторная база данных FAISS успешно подключена!")
else:
    print(f"Ошибка: Папка {DB_DIR} не найдена. Сначала запустите create_db.py!")
    exit()


# === ЛОГИКА ОБРАБОТКИ КОМАНД ===

# Ответ на команду /start
@bot.message_handler(commands=['start'])
def send_welcome(message):
    welcome_text = (
        "Приветствую! Я твой цифровой AI-Ассистент инженера.\n"
        "Я полностью изучил мануал по ремонту. Напиши мне код ошибки или "
        "описание поломки, и я найду точную инструкцию по ремонту."
    )
    bot.reply_to(message, welcome_text)

# Обработка любого текстового вопроса от пользователя
@bot.message_handler(content_types=['text'])
def handle_message(message):
    user_query = message.text
    bot.send_chat_action(message.chat.id, 'typing') # Визуальный эффект "бот печатает"

    print(f"Получен запрос от пользователя: {user_query}")
    
    # Шаг A: Ищем в мануале 3 самых подходящих текстовых кусочка
    docs = db.similarity_search(user_query, k=3)
    
    # Собираем найденные фрагменты в один блок контекста
    context = "\n---\n".join([doc.page_content for doc in docs])
    
    # Шаг B: Формируем системный промт с жесткой ролью и рамками (Промт-инжиниринг)
    system_prompt = (
        "Ты — ведущий шеф-инженер по ремонту сложной техники, автоэлектрики и КИПиА. "
        "К тебе за помощью обратился техник. Твоя задача — дать четкий ответ на основе "
        "предоставленного технического руководства (мануала).\n\n"
        "ПРАВИЛА ОТВЕТА:\n"
        "1. Отвечай строго по фактам из документа.\n"
        "2. Структурируй ответ: возможная причина поломки, пошаговый план диагностики, "
        "инструменты и меры безопасности.\n"
        "3. Если в мануале нет информации по этому вопросу, честно ответь, что "
        "в руководстве этого нет.\n\n"
        f"ВЫРЕЗКИ ИЗ МАНУАЛА:\n{context}"
    )

    # Шаг C: Отправляем запрос в GigaChat через API
    try:
        with GigaChat(credentials=GIGACHAT_CREDENTIALS, verify_ssl_certs=False) as giga:
            response = giga.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ]
            )
            bot_reply = response.choices.message.content
    except Exception as e:
        print(f"Ошибка API GigaChat: {e}")
        bot_reply = "Извините, произошла техническая ошибка при обращении к ИИ. Попробуйте позже."

    # Отправляем ответ пользователю в Telegram
    bot.reply_to(message, bot_reply)

# Запуск постоянного прослушивания сообщений
print("Бот успешно запущен и готов к работе в Telegram!")
bot.infinity_polling()

