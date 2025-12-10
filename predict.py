import numpy as np
from src.preprocessing import TextCleaner, EmojiProcessor, SlangNormalizer, TextTokenizer
from src.model import SentimentLSTMModel
from src.utils import config

def predict_sentiment(texts, model_path, tokenizer_path):
    cleaner = TextCleaner()
    emoji_proc = EmojiProcessor()
    slang_norm = SlangNormalizer()
    
    processed = []
    for text in texts:
        text = emoji_proc.process_text(text)
        text = cleaner.clean_text(text)
        text = slang_norm.normalize_text(text)
        processed.append(text)
    
    tokenizer = TextTokenizer(config.VOCAB_SIZE, config.MAX_SEQUENCE_LENGTH)
    tokenizer.load_tokenizer(tokenizer_path)
    
    sequences = tokenizer.texts_to_sequences(processed)
    
    model_arch = SentimentLSTMModel(
        vocab_size=config.VOCAB_SIZE,
        embedding_dim=config.EMBEDDING_DIM,
        lstm_units=config.LSTM_UNITS,
        dense_units=config.DENSE_UNITS,
        dropout_rate=config.DROPOUT_RATE,
        num_classes=config.NUM_CLASSES,
        max_length=config.MAX_SEQUENCE_LENGTH
    )
    
    model_arch.load_model(model_path)
    
    predictions = model_arch.model.predict(sequences)
    predicted_classes = predictions.argmax(axis=1)
    
    results = []
    for i, text in enumerate(texts):
        results.append({
            'text': text,
            'sentiment': config.CLASS_NAMES[predicted_classes[i]],
            'confidence': predictions[i][predicted_classes[i]]
        })
    
    return results

def main():
    test_texts = [
        "I love this product! It's amazing 😍",
        "This is terrible. Worst experience ever 😡",
        "It's okay, nothing special",
        "omg this is so good lol 👍",
        "Not impressed tbh"
    ]
    
    print("Loading model and making predictions...\n")
    
    results = predict_sentiment(
        test_texts,
        config.MODEL_SAVE_PATH,
        'data/processed/tokenizer.pkl'
    )
    
    for i, result in enumerate(results, 1):
        print(f"{i}. Text: {result['text']}")
        print(f"   Sentiment: {result['sentiment']}")
        print(f"   Confidence: {result['confidence']:.4f}\n")

if __name__ == "__main__":
    main()
