import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from src.preprocessing import TextCleaner, EmojiProcessor, SlangNormalizer, TextTokenizer
from src.model import SentimentLSTMModel, ModelTrainer
from src.evaluation import MetricsEvaluator
from src.utils import config

def train_baseline_models(X_train, y_train, X_test, y_test):
    print("\nTraining baseline models...")
    
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    results = {}
    evaluator = MetricsEvaluator(config.CLASS_NAMES)
    
    print("\n1. Naive Bayes...")
    nb = MultinomialNB()
    nb.fit(X_train_tfidf, y_train)
    y_pred_nb = nb.predict(X_test_tfidf)
    results['Naive Bayes'] = evaluator.calculate_metrics(y_test, y_pred_nb)
    
    print("2. Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_tfidf, y_train)
    y_pred_lr = lr.predict(X_test_tfidf)
    results['Logistic Regression'] = evaluator.calculate_metrics(y_test, y_pred_lr)
    
    print("3. SVM...")
    svm = SVC(kernel='rbf', random_state=42)
    svm.fit(X_train_tfidf, y_train)
    y_pred_svm = svm.predict(X_test_tfidf)
    results['SVM'] = evaluator.calculate_metrics(y_test, y_pred_svm)
    
    return results

def train_lstm_model(texts, labels, use_emoji_slang=False):
    cleaner = TextCleaner()
    emoji_proc = EmojiProcessor()
    slang_norm = SlangNormalizer()
    
    processed = []
    for text in texts:
        if use_emoji_slang:
            text = emoji_proc.process_text(text)
        text = cleaner.clean_text(text)
        if use_emoji_slang:
            text = slang_norm.normalize_text(text)
        processed.append(text)
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        processed, labels, test_size=config.TEST_SPLIT, random_state=42, stratify=labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=config.VAL_SPLIT/(config.TRAIN_SPLIT + config.VAL_SPLIT),
        random_state=42, stratify=y_train_val
    )
    
    tokenizer = TextTokenizer(config.VOCAB_SIZE, config.MAX_SEQUENCE_LENGTH)
    tokenizer.fit_on_texts(X_train)
    
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_val_seq = tokenizer.texts_to_sequences(X_val)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    model_arch = SentimentLSTMModel(
        vocab_size=config.VOCAB_SIZE,
        embedding_dim=config.EMBEDDING_DIM,
        lstm_units=config.LSTM_UNITS,
        dense_units=config.DENSE_UNITS,
        dropout_rate=config.DROPOUT_RATE,
        num_classes=config.NUM_CLASSES,
        max_length=config.MAX_SEQUENCE_LENGTH
    )
    
    model = model_arch.build_model()
    model_arch.compile_model(config.LEARNING_RATE)
    
    trainer = ModelTrainer(model)
    trainer.train(X_train_seq, y_train, X_val_seq, y_val,
                 batch_size=config.BATCH_SIZE, epochs=config.EPOCHS,
                 patience=config.PATIENCE, save_path='models/saved/temp_model.h5')
    
    y_pred = trainer.predict(X_test_seq)
    
    evaluator = MetricsEvaluator(config.CLASS_NAMES)
    metrics = evaluator.calculate_metrics(y_test, y_pred)
    
    return metrics, X_train, X_test, y_test

def compare_models():
    print(f"\n{'='*70}")
    print("MODEL COMPARISON")
    print(f"{'='*70}\n")
    
    df = pd.read_csv(config.DATA_PATH)
    texts = df['text'].values
    labels = df['sentiment'].values
    
    print("Training LSTM (baseline - no emoji/slang processing)...")
    lstm_basic_metrics, X_train, X_test, y_test = train_lstm_model(texts, labels, use_emoji_slang=False)
    
    print("\nTraining LSTM (modified - with emoji/slang processing)...")
    lstm_modified_metrics, _, _, _ = train_lstm_model(texts, labels, use_emoji_slang=True)
    
    baseline_results = train_baseline_models(X_train, y_train, X_test, y_test)
    
    print(f"\n{'='*70}")
    print("COMPARISON RESULTS")
    print(f"{'='*70}\n")
    
    all_results = {
        'Naive Bayes': baseline_results['Naive Bayes'],
        'Logistic Regression': baseline_results['Logistic Regression'],
        'SVM': baseline_results['SVM'],
        'LSTM (Basic)': lstm_basic_metrics,
        'LSTM (Modified)': lstm_modified_metrics
    }
    
    print(f"{'Model':<25} {'Accuracy':<12} {'Macro F1':<12} {'Weighted F1':<12}")
    print(f"{'-'*70}")
    
    for model_name, metrics in all_results.items():
        print(f"{model_name:<25} {metrics['accuracy']:<12.4f} "
              f"{metrics['macro_f1']:<12.4f} {metrics['weighted_f1']:<12.4f}")
    
    print(f"\n{'='*70}")

if __name__ == "__main__":
    compare_models()
