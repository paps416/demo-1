from data_loader import load_imdb_data
from text_preprocessor import normalize_text
from models import train_tfidf_classifier, train_bow_classifier, evaluate_model

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
    
    print("\nмодель TF_IDF")
    tfidf_model, tfidf_vectorizer = train_tfidf_classifier(processed_train_texts, train_labels)
    tfidf_acc, tfidf_f1 = evaluate_model(tfidf_model, tfidf_vectorizer, processed_test_texts, test_labels)
    print(f"результат F1-метрики для TF-IDF: {tfidf_f1:.4f}")
    
    print("\nфинальные результаты")
    print(f"Train samples: {TRAIN_SAMPLES}, Test samples: {TEST_SAMPLES}\n")
    print(f"BoW Model Accuracy: {bow_acc:.2f}")
    print(f"BoW Model F1-Score: {bow_f1:.2f}\n")
    print(f"TF-IDF Model Accuracy: {tfidf_acc:.2f}")
    print(f"TF-IDF Model F1-Score: {tfidf_f1:.2f}")