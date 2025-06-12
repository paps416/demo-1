import spacy
import emoji  # extra


def normalize_text(text):
    """
    выполняет нормализацию текста (перевод в нижний регистр + лемматизацию + удаление
    стоп-слов + знаков препинания и пробелов

    Args:
        text (str): обычная входная строка

    Returns:
        str: очищенная строка
    """

    # emoji to text
    text = emoji.demojize(text, language="alias")

    # spacy load
    try:
        # using smallest model for speed + pc power issues
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("модель не найдена")
        return ""

    # filter
    doc = nlp(text.lower())

    # empty list
    clean_words = []

    # working on every word
    for token in doc:
        if not token.is_stop and not token.is_punct and not token.is_space:
            clean_words.append(token.lemma_)

    return " ".join(clean_words)
