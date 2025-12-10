import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

from src.preprocessing import TextCleaner, EmojiProcessor, SlangNormalizer, TextTokenizer
from src.model import SentimentLSTMModel, ModelTrainer
from src.evaluation import MetricsEvaluator
from src.utils import config

def load_data():
    print("Loading dataset...")
    df = pd.read_csv(config.DATA_PATH)
    texts = df['text'].values
    labels = df['sentiment'].values
    return texts, labels

def preprocess_data(texts, use_emoji_slang=True):
    print("Preprocessing texts...")
    cleaner = TextCleaner()
    emoji_proc = EmojiProcessor()
    slang_norm = SlangNormalizer()
    
    processed_texts = []
    for text in texts:
        if use_emoji_slang:
            text = emoji_proc.process_text(text)
        text = cleaner.clean_text(text)
        if use_emoji_slang:
            text = slang_norm.normalize_text(text)
        processed_texts.append(text)
    
    return processed_texts

def prepare_sequences(texts, labels):
    print("Tokenizing and creating sequences...")
    tokenizer = TextTokenizer(config.VOCAB_SIZE, config.MAX_SEQUENCE_LENGTH)
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        texts, labels, test_size=config.TEST_SPLIT, random_state=42, stratify=labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, 
        test_size=config.VAL_SPLIT/(config.TRAIN_SPLIT + config.VAL_SPLIT),
        random_state=42, stratify=y_train_val
    )
    
    tokenizer.fit_on_texts(X_train)
    
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_val_seq = tokenizer.texts_to_sequences(X_val)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    os.makedirs(config.PROCESSED_DATA_PATH, exist_ok=True)
    tokenizer.save_tokenizer(os.path.join(config.PROCESSED_DATA_PATH, 'tokenizer.pkl'))
    
    return X_train_seq, X_val_seq, X_test_seq, y_train, y_val, y_test, tokenizer

def build_and_train_model(X_train, y_train, X_val, y_val, word_index):
    print("Building model...")
    model_arch = SentimentLSTMModel(
        vocab_size=config.VOCAB_SIZE,
        embedding_dim=config.EMBEDDING_DIM,
        lstm_units=config.LSTM_UNITS,
        dense_units=config.DENSE_UNITS,
        dropout_rate=config.DROPOUT_RATE,
        num_classes=config.NUM_CLASSES,
        max_length=config.MAX_SEQUENCE_LENGTH
    )
    
    model = model_arch.build_model(word_index, config.WORD2VEC_PATH)
    model_arch.compile_model(config.LEARNING_RATE)
    
    print("\nModel Summary:")
    model.summary()
    
    print("\nTraining model...")
    trainer = ModelTrainer(model)
    
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
    
    history = trainer.train(
        X_train, y_train, X_val, y_val,
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        patience=config.PATIENCE,
        save_path=config.MODEL_SAVE_PATH
    )
    
    return trainer, history

def evaluate_model(trainer, X_test, y_test):
    print("\nEvaluating model on test set...")
    y_pred = trainer.predict(X_test)
    
    evaluator = MetricsEvaluator(config.CLASS_NAMES)
    metrics = evaluator.calculate_metrics(y_test, y_pred)
    cm = evaluator.get_confusion_matrix(y_test, y_pred)
    
    evaluator.print_metrics(metrics)
    evaluator.print_confusion_matrix(cm)
    
    return metrics

def main():
    print(f"\n{'='*70}")
    print("SENTIMENT CLASSIFICATION - LSTM MODEL")
    print(f"{'='*70}\n")
    
    texts, labels = load_data()
    print(f"Dataset size: {len(texts)} samples")
    print(f"Class distribution: {np.bincount(labels)}")
    
    processed_texts = preprocess_data(texts, use_emoji_slang=True)
    
    X_train, X_val, X_test, y_train, y_val, y_test, tokenizer = prepare_sequences(
        processed_texts, labels
    )
    
    print(f"\nTrain set: {len(X_train)}")
    print(f"Validation set: {len(X_val)}")
    print(f"Test set: {len(X_test)}")
    
    trainer, history = build_and_train_model(
        X_train, y_train, X_val, y_val,
        tokenizer.tokenizer.word_index
    )
    
    metrics = evaluate_model(trainer, X_test, y_test)
    
    print(f"\n{'='*70}")
    print("Training completed successfully!")
    print(f"Model saved to: {config.MODEL_SAVE_PATH}")
    print(f"{'='*70}\n")
    
    return trainer, metrics

if __name__ == "__main__":
    main()
