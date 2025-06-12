import spacy
# import emoji # extra

def normalize_text(text):
    
    # emoji to text
    # text = emoji.demojize(text, language='alias')
    
    # spacy load
    try:
        nlp = spacy.load('en_core_web_sm') # using smallest model for speed + pc power issues
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