import pandas as pd
import numpy as np
import re
from models import CharTokenizer

if __name__ == "__main__":
    import os
    # Загрузка датасета с поисковыми запросами
    splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
    df_train = pd.read_parquet("hf://datasets/EmbeddingStudio/synthetic-search-queries-ru/" + splits["train"])
    df_val = pd.read_parquet("hf://datasets/EmbeddingStudio/synthetic-search-queries-ru/" + splits["test"])

    for df in (df_train, df_val):
        # Удаление всех пробелов
        df['text'] = df['Query'].apply(lambda x: re.sub(r'\s+', ' ', x))
        df['text_no_spaces'] = df['text'].apply(lambda x: ''.join(x.split()))
    
    # Используем символьный токенизатор
    tokenizer = CharTokenizer()

    # Функция для токенизации и разметки
    def tokenize_and_label(row):
        text = row['text']  # Original text with spaces
        text_no_spaces = row['text_no_spaces']  # Text without spaces
        
        if not isinstance(text, str) or not isinstance(text_no_spaces, str) or not text_no_spaces:
            return [], [], ""
        
        # Tokenize the text without spaces (each character becomes a token)
        tokens = tokenizer.tokenize(text_no_spaces)
        labels = []
        
        # Create a mapping to determine where spaces should be
        text_chars = [char for char in text if char != ' ']  # Characters without spaces
        space_positions = set()  # Positions where spaces should be after characters
        
        # Find positions where spaces should be inserted
        char_idx = 0
        for i, char in enumerate(text):
            if char == ' ':
                # Mark that there should be a space after the previous character
                if char_idx > 0:
                    space_positions.add(char_idx - 1)
            else:
                char_idx += 1
        
        # Create labels: 1 if space should follow this character, 0 otherwise
        for i, token in enumerate(tokens):
            if i in space_positions:
                labels.append(1)
            else:
                labels.append(0)
        
        # Reconstruct text using tokens and labels
        remade_text = ""
        for i, token in enumerate(tokens):
            remade_text += token
            if i < len(labels) and labels[i] == 1:
                remade_text += ' '
        
        return tokens, labels, remade_text

    # Применить функцию к датасетам
    for df in (df_train, df_val):
        df['tokens'], df['labels'], df['remade_text'] = zip(*df.apply(tokenize_and_label, axis=1))

    # Ensure output directory exists
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(output_dir, exist_ok=True)

    # Сохранить датасеты
    df_train_out = df_train[['text_no_spaces', 'tokens', 'labels', 'text', 'remade_text']]
    df_val_out = df_val[['text_no_spaces', 'tokens', 'labels', 'text', 'remade_text']]
    df_train_out.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    df_val_out.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
