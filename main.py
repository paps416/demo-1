from data_loader import load_imdb_data
from text_preprocessor import normalize_text
from models import train_tfidf_classifier, train_bow_classifier, evaluate_model
from tqdm import tqdm
import spacy


def console_interface(model, vectorizer):
    """
    простой консольный интерфейс для анализа отзывов
    """
    print("\nанализ кастомного отзыва")
    print("введите текст для анализа - для выхода напишите выход/exit/quit/q")

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("модель не найдена")
        nlp = None

    while True:
        user_input = input("\n> ")
        if user_input.lower() in ["выход", "exit", "quit", "q"]:
            print("o7")
            break

        processed_text = normalize_text(user_input)
        vectorized_text = vectorizer.transform([processed_text])
        prediction = model.predict(vectorized_text)
        sentiment = "позитивная" if prediction[0] == 1 else "негативная"

        print(f"тональность: {sentiment}")

        if nlp:
            doc = nlp(user_input)
            aspects = [chunk.text for chunk in doc.noun_chunks]
            if aspects:
                print(f"выделенные аспекты: {', '.join(aspects)}")
            else:
                print("выделенные аспекты: не найдены.")


if __name__ == "__main__":
    # settings
    TRAIN_SAMPLES = 10000
    TEST_SAMPLES = 2500

    print(f"загружаем {TRAIN_SAMPLES} обучающих и {TEST_SAMPLES} тестовых сэмплов")
    (train_texts, train_labels), (test_texts, test_labels) = load_imdb_data(
        train_samples=TRAIN_SAMPLES, test_samples=TEST_SAMPLES
    )

    print("\nначинаем предобработку")

    # map -> tqdm
    processed_train_texts = list(
        tqdm(
            map(normalize_text, train_texts),
            total=len(train_texts),
            desc="обработка обучающих текстов  ",
        )
    )

    processed_test_texts = list(
        tqdm(
            map(normalize_text, test_texts),
            total=len(test_texts),
            desc="обработка тестовых текстов  ",
        )
    )

    print("предобработка завершена")

    print("\nмодель bow")
    bow_model, bow_vectorizer = train_bow_classifier(
        processed_train_texts, train_labels
    )
    bow_acc, bow_f1 = evaluate_model(
        bow_model, bow_vectorizer, processed_test_texts, test_labels
    )
    print(f"результат F1-метрики для bow: {bow_f1:.4f}")

    print("\nмодель TF_IDF")
    tfidf_model, tfidf_vectorizer = train_tfidf_classifier(
        processed_train_texts, train_labels
    )
    tfidf_acc, tfidf_f1 = evaluate_model(
        tfidf_model, tfidf_vectorizer, processed_test_texts, test_labels
    )
    print(f"результат F1-метрики для TF-IDF: {tfidf_f1:.4f}")

    print("\n**финальные результаты**")
    print(f"Train samples: {TRAIN_SAMPLES}, Test samples: {TEST_SAMPLES}\n")
    print(f"BoW Model Accuracy: {bow_acc:.2f}")
    print(f"BoW Model F1-Score: {bow_f1:.2f}\n")
    print(f"TF-IDF Model Accuracy: {tfidf_acc:.2f}")
    print(f"TF-IDF Model F1-Score: {tfidf_f1:.2f}")

    console_interface(tfidf_model, tfidf_vectorizer)
