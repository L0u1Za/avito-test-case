import pandas as pd
import numpy as np
from transformers import AutoTokenizer
import re

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
    
    # Используем предобученный токенизатор
    tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")

    # Функция для токенизации и разметки
    def tokenize_and_label(row):
        text = row['text']
        text_no_spaces = row['text_no_spaces']
        if not isinstance(text, str) or not isinstance(text_no_spaces, str) or not text_no_spaces:
            return [], []
        tokens = tokenizer.tokenize(text_no_spaces)
        labels = []
        idx_in_text = 0
        for token in tokens:
            cur_token_no_special_chars = token.replace('##', '')
            cur_text = ''
            while idx_in_text < len(text):
                if text[idx_in_text] != ' ':
                    cur_text += text[idx_in_text]
                if cur_text == cur_token_no_special_chars:
                    break
                idx_in_text += 1
            idx_in_text += 1
            if idx_in_text < len(text) and text[idx_in_text] == ' ':
                labels.append(1)
            else:
                labels.append(0)

        remade_text = ""
        for (i, token) in enumerate(tokens):
            tok = token.replace('##', '')
            if labels[i] == 1:
                tok += ' '
            remade_text += tok
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
