from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
# import joblib

def train_tfidf_classifier(texts, labels):
    print("TF-IDF in process")
    return None, None

def train_bow_classifier(texts, labels):

    print("обучаем модель на основе bow")
    
    vectorizer = CountVectorizer(
        ngram_range=(1, 2), 
        max_features=25000
    )
    X = vectorizer.fit_transform(texts)
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, labels)
    
    return model, vectorizer

def evaluate_model(model, vectorizer, texts, labels):
    X_test = vectorizer.transform(texts)
    predictions = model.predict(X_test)
    
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='binary') 
    
    return acc, f1