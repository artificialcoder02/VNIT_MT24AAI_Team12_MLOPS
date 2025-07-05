import os
import pandas as pd

def load_reviews_from_folder(folder_path, sentiment):
    reviews = []
    for filename in os.listdir(folder_path):
        with open(os.path.join(folder_path, filename), encoding='utf-8') as f:
            content = f.read()
            reviews.append((content, sentiment))
    return reviews

def build_imdb_csv(dataset_path='data/aclImdb'):
    data = []

    for set_name in ['train', 'test']:
        for sentiment in ['pos', 'neg']:
            path = os.path.join(dataset_path, set_name, sentiment)
            print(f"Loading from {path}...")
            sentiment_label = 'positive' if sentiment == 'pos' else 'negative'
            reviews = load_reviews_from_folder(path, sentiment_label)
            data.extend(reviews)

    df = pd.DataFrame(data, columns=['review', 'sentiment'])
    df.to_csv('data/imdb_reviews.csv', index=False)
    print("✅ Dataset converted to data/imdb_reviews.csv")

if __name__ == "__main__":
    build_imdb_csv()
