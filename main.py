from data_loader import load_imdb_data

if __name__ == "__main__":
    # settings
    TRAIN_SAMPLES = 1000
    TEST_SAMPLES = 250

    print(f"загружаем {TRAIN_SAMPLES} обучающих и {TEST_SAMPLES} тестовых сэмплов")
    (train_texts, train_labels), (test_texts, test_labels) = load_imdb_data(
        train_samples=TRAIN_SAMPLES,
        test_samples=TEST_SAMPLES
    )

    print("!!")