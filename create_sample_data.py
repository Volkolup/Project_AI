import pandas as pd
import numpy as np

sample_data = {
    'text': [
        "I love this product! It's amazing 😍",
        "This is terrible. Worst experience ever 😡",
        "It's okay, nothing special",
        "Absolutely fantastic! Best purchase ever 👍",
        "Horrible service, very disappointed 😢",
        "Average quality, meets expectations",
        "So happy with this! Highly recommend 😊",
        "Waste of money. Don't buy this 👎",
        "It works fine, no complaints",
        "Outstanding! Exceeded my expectations ⭐",
    ],
    'sentiment': [2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
}

df = pd.DataFrame(sample_data)
df.to_csv('data/raw/sample_dataset.csv', index=False)
print("Sample dataset created: data/raw/sample_dataset.csv")
print(f"Total samples: {len(df)}")
print(f"Class distribution:\n{df['sentiment'].value_counts().sort_index()}")
