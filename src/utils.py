import pandas as pd
import csv
import re

def preprocess_dataset(path):
    rows = []
    with open('../../data/dataset_1937770_3.txt', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            # split only on first comma:
            id_part, rest = line.split(',', 1)
            rows.append([id_part, rest])

    df = pd.DataFrame(rows[1:], columns=rows[0])
    return df

def get_space_positions(text):
    """
    Возвращает множество индексов (позиций) в строке text,
    после которых должен стоять пробел.
    Позиции отсчитываются по символам, 0-based.
    Например, для "книга в хорошем" будет positions = {5, 7, ...} и т.д.
    """
    positions = set()
    cur_pos = 0
    # идём по символам, и если после текущего символа в оригинале есть пробел — записываем позицию
    for i in range(len(text) - 1):
        if text[i+1] == " ":
            positions.add(cur_pos)
        else:
            cur_pos += 1
    return positions

def f1_for_one(true_text, pred_text):
    """
    Вычисляет F1-score для одной строки.
    true_text и pred_text — строки с пробелами в нужных местах.
    Предполагается, что оба текста выровнены по содержимому (т.е. непробельные символы в порядке совпадают).
    """
    true_pos = get_space_positions(true_text)
    pred_pos = get_space_positions(pred_text)
    
    if not true_pos and not pred_pos:
        # если в истине нет пробелов и предсказание тоже без пробелов,
        return 1.0
    if not true_pos and pred_pos:
        # истина без пробелов, предсказание ставит — вся предсказанная разбивка ошибочна
        return 0.0
    if true_pos and not pred_pos:
        # истина содержит пробелы, а предсказание — нет
        return 0.0
    
    tp = len(true_pos & pred_pos)
    fp = len(pred_pos - true_pos)
    fn = len(true_pos - pred_pos)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def average_f1(true_texts, pred_texts):
    """
    Возвращает средний F1 по всем строкам.
    """
    assert len(true_texts) == len(pred_texts), "Списки должны быть одинаковой длинны"
    total = 0.0
    n = len(true_texts)
    for t, p in zip(true_texts, pred_texts):
        total += f1_for_one(t, p)
    return total / n

def post_process_text(text: str) -> str:
    """
    Удаляет лишние пробелы из текста:
    - убирает ведущие и конечные пробелы
    - заменяет несколько подряд идущих пробелов на один
    """
    if text is None:
        return text

    s = text.strip()
    s = re.sub(r'\s+', ' ', s)
    return s

# Пример использования
if __name__ == "__main__":
    true = [
        "книга в хорошем состоянии",
        "ищу дом в Подмосковье",
        "отдам даром кошку"
    ]
    pred = [
        "книга вхорошем состоянии",
        "ищудом вПодмосковье",
        "от дам даромкошку"
    ]
    print(average_f1(true, pred))