import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Новый современный импорт для работы с локальными текстовыми моделями
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

PDF_PATH = "manual.pdf"      
DB_DIR = "faiss_index"        

def main():
    if not os.path.exists(PDF_PATH):
        print(f"Ошибка: Файл {PDF_PATH} не найден! Положите мануал в папку проекта.")
        return

    print("1. Загрузка PDF и чтение текста (для 150 МБ это займет около 1-3 минут)...")
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    print(f"Успешно прочитано: {len(docs)} страниц.")

    print("\n2. Разрезка текста на смысловые фрагменты (чанги)...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(f"Создано фрагментов текста: {len(splits)} шт.")

    print("\n3. Инициализация ИИ-модели эмбеддингов (обработка локально на ПК)...")
    # Используем обновленный класс HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    print("\n4. Индексация текста и создание векторной базы данных FAISS...")
    db = FAISS.from_documents(splits, embeddings)
    
    print(f"\n5. Сохранение базы данных в папку '{DB_DIR}'...")
    db.save_local(DB_DIR)
    print("Процесс успешно завершен! База данных готова к работе.")

if __name__ == "__main__":
    main()
