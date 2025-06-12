from data_loader import load_imdb_data
from text_preprocessor import normalize_text
from models import train_bow_classifier, evaluate_model

if __name__ == "__main__":
    # settings
    TRAIN_SAMPLES = 100
    TEST_SAMPLES = 25

    print(f"загружаем {TRAIN_SAMPLES} обучающих и {TEST_SAMPLES} тестовых сэмплов")
    (train_texts, train_labels), (test_texts, test_labels) = load_imdb_data(
        train_samples=TRAIN_SAMPLES,
        test_samples=TEST_SAMPLES
    )

    print("\nначинаем предобработку")
    
    processed_train_texts = [normalize_text(text) for text in train_texts]
    processed_test_texts = [normalize_text(text) for text in test_texts]
    
    print("предобработка завершена")
    
    print("\nмодель bow")
    bow_model, bow_vectorizer = train_bow_classifier(processed_train_texts, train_labels)
    bow_acc, bow_f1 = evaluate_model(bow_model, bow_vectorizer, processed_test_texts, test_labels)
    print(f"результат F1-метрики для bow: {bow_f1:.4f}")