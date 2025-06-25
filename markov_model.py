
import random
import pandas as pd
from collections import defaultdict

def build_transition_matrix(data):
    matrix = defaultdict(lambda: defaultdict(int))
    for number in data:
        digits = f"{int(number):04d}"
        for i in range(3):
            matrix[digits[i]][digits[i+1]] += 1
    return matrix

def generate_next_digit(matrix, current):
    if current not in matrix or not matrix[current]:
        return str(random.randint(0, 9))
    probs = matrix[current]
    total = sum(probs.values())
    rand = random.randint(1, total)
    cumulative = 0
    for digit, count in probs.items():
        cumulative += count
        if rand <= cumulative:
            return digit
    return str(random.randint(0, 9))

def prediksi_markov(df):
    data = df["angka"].astype(str).tolist()
    matrix = build_transition_matrix(data)
    prediksi = str(random.randint(0, 9))
    for _ in range(3):
        next_d = generate_next_digit(matrix, prediksi[-1])
        prediksi += next_d
    return prediksi
