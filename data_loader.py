from datasets import load_dataset


def load_imdb_data(train_samples=2000, test_samples=500):
    """
    загружает и подготавливает датасет

    Args:
        train_samples (int): кол во train samples
        test_samples (int): кол во test samples

    Returns:
        ((train_texts, train_labels), (test_texts, test_labels)).
    """

    print("IMBD дата сет загружается")

    imdb_dataset = load_dataset("imdb")

    # shuffle for randomness of labels
    train_data = imdb_dataset["train"].shuffle(seed=42).select(range(train_samples))
    test_data = imdb_dataset["test"].shuffle(seed=42).select(range(test_samples))

    # 0 - neg, 1 - pos
    train_texts = [item["text"] for item in train_data]
    train_labels = [item["label"] for item in train_data]

    test_texts = [item["text"] for item in test_data]
    test_labels = [item["label"] for item in test_data]

    print("IMDb дата сет загружен")
    return (train_texts, train_labels), (test_texts, test_labels)
