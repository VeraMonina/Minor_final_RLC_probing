# -*- coding: utf-8 -*-
"""
03_probing.py
-------------
Обучаем пробинг-классификаторы и считаем метрики для всех задач.

Задачи на уровне токенов (морфология + позиция в предложении):
    Case, Number, Aspect, Tense, Gender, Person, Mood, position_bin

Задачи на уровне предложений (синтаксис):
    sent_len_bin   — длина предложения (short / medium / long)
    subj_position  — позиция подлежащего относительно сказуемого (first / non-first)
    verb_position  — позиция сказуемого (early / middle / late)
    svo_order      — порядок слов (SVO / SOV / ...)

Контрольное условие:
    random_baseline — те же метки, но на случайных эмбеддингах той же формы

Сохраняемые метрики (в results/probing_results.csv):
    acc_mean / acc_std    — точность по 3-кратной кросс-валидации
    f1_mean  / f1_std     — макро-F1
    selectivity           — acc_mean - random_baseline_acc  
    majority              — доля самого частотного класса

Важно про индексы и .npy файлы:
    index_{model}.csv       — метаданные, одна строка на токен
    index_sent_{model}.csv  — метаданные, одна строка на предложение
    {model}_layer{N}.npy    — эмбеддинги, форма (число_строк, 768)
    Строка i в CSV соответствует строке i в каждом .npy для этой модели.
    Маска mask_arr выбирает одни и те же строки из обоих.

Оптимизация:
    Каждый файл слоя загружается ОДИН раз на задачу, а потом для всех L1-групп
    берутся нужные срезы. Вместо N_LAYERS * N_L1s загрузок делаем N_LAYERS.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

emb_dir = Path("embeddings")
results_dir = Path("results")
plots_dir = results_dir / "plots"
results_dir.mkdir(exist_ok=True)
plots_dir.mkdir(exist_ok=True)

models_list = ["rubert", "mbert", "xlmr"]   # какие модели обрабатываем
n_layers = 13          # всего слоёв (0..12)
n_folds = 3            # число фолдов кросс-валидации
min_samples_per_class = 20   # минимальное количество примеров в классе, чтобы включать в анализ
lr_params = {"C": 1.0, "max_iter": 2000, "random_state": 42}


tasks = {
    "Aspect":       {"level": "token", "col": "aspect",       "upos": {"VERB","AUX"}}, # прописываем еще раз задачу, уровень на котором она разбирается и часторечные категории для которых она актуально
    "Tense":        {"level": "token", "col": "tense",        "upos": {"VERB","AUX"}},
    "Gender":       {"level": "token", "col": "gender",       "upos": {"NOUN","ADJ","VERB"}},
    "Case":         {"level": "token", "col": "case",         "upos": {"NOUN","PRON","ADJ","DET","NUM"}},
    "Number":       {"level": "token", "col": "number",       "upos": {"NOUN","PRON","ADJ","DET","NUM","VERB"}},
    "Person":       {"level": "token", "col": "person",       "upos": {"VERB","AUX","PRON"}},
    "Mood":         {"level": "token", "col": "mood",         "upos": {"VERB","AUX"}},
    "position_bin": {"level": "token", "col": "position_bin", "upos": None},
    "sent_len_bin":  {"level": "sent", "col": "sent_len_bin",  "upos": None},
    "subj_position": {"level": "sent", "col": "subj_position", "upos": None},
    "verb_position": {"level": "sent", "col": "verb_position", "upos": None},
    "svo_order":     {"level": "sent", "col": "svo_order",     "upos": None},
}


def probe(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Обучает логистическую регрессию с кросс-валидацией на эмбеддингах X и метках y.
    Возвращает среднюю точность и макро-F1 по фолдам.
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    accs, f1s = [], []
    for train_idx, test_idx in cv.split(X, y):
        clf = LogisticRegression(**lr_params)
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[test_idx])
        accs.append(accuracy_score(y[test_idx], preds))
        f1s.append(f1_score(y[test_idx], preds, average="macro", zero_division=0))
    return {
        "acc_mean": float(np.mean(accs)),
        "acc_std":  float(np.std(accs)),
        "f1_mean":  float(np.mean(f1s)),
        "f1_std":   float(np.std(f1s)),
    }

def probe_random(X: np.ndarray, y: np.ndarray) -> dict:
    """Тот же пробинг, но на случайных эмбеддингах той же формы."""
    X_rand = np.random.default_rng(42).standard_normal(X.shape).astype(np.float32)
    return probe(X_rand, y)

def majority_baseline(y: np.ndarray) -> float:
    """Доля самого частотного класса."""
    _, counts = np.unique(y, return_counts=True)
    return float(counts.max() / len(y))

def check_group_size(y: np.ndarray) -> bool:
    """Проверяет, что есть хотя бы два класса и минимум примеров на класс."""
    _, counts = np.unique(y, return_counts=True)
    return len(counts) >= 2 and int(counts.min()) >= min_samples_per_class


print("Запускаем пробинг-эксперименты...")

all_results = []

for model_alias in models_list:
    print(f"\nМОДЕЛЬ: {model_alias}")

    index_token = pd.read_csv(emb_dir / f"index_{model_alias}.csv")
    index_sent  = pd.read_csv(emb_dir / f"index_sent_{model_alias}.csv")
    index_token.columns = index_token.columns.str.lower()
    index_sent.columns  = index_sent.columns.str.lower()

    for task_name, task_cfg in tasks.items():
        level = task_cfg["level"]
        col   = task_cfg["col"]
        upos  = task_cfg["upos"]

        print(f"\n Задача: {task_name}  (уровень={level})")

        index = index_token if level == "token" else index_sent
        emb_prefix = model_alias if level == "token" else f"sent_{model_alias}"

        # Маска для отбора строк с непустым значением признака
        mask = index[col].notna()
        if upos is not None and "upos" in index.columns:
            mask = mask & (index["upos"].isin(upos))

        task_index = index[mask].reset_index(drop=True)
        mask_arr = mask.values

        if len(task_index) < min_samples_per_class * 2:
            print(f"    Недостаточно данных ({len(task_index)}) — пропускаем")
            continue

        le = LabelEncoder()
        y_all = le.fit_transform(task_index[col].values)
        print(f"    Классы: {list(le.classes_)}  |  примеров: {len(task_index):,}")

        majority = majority_baseline(y_all)
        print(f"    Majority baseline: {majority:.3f}")

        # Группировка по L1
        l1_groups = {l1: grp.index.values for l1, grp in task_index.groupby("dominant_language")}
        # Группировка по ресурсности
        resource_groups = {}
        if "resource_type" in task_index.columns:
            resource_groups = {rt: grp.index.values for rt, grp in task_index.groupby("resource_type")}

        # Random baseline на слое 12
        rand_acc = None
        last_emb_path = emb_dir / f"{emb_prefix}_layer12.npy"
        if last_emb_path.exists():
            X_last = np.load(last_emb_path)[mask_arr]
            rand_res = probe_random(X_last, y_all)
            rand_acc = rand_res["acc_mean"]
            del X_last

            all_results.append({
                "model":          model_alias,
                "task":           task_name,
                "level":          level,
                "l1":             "random_baseline",
                "resource_type":  "n/a",
                "layer":          12,
                "n_items":        len(y_all),
                "majority":       round(majority, 4),
                "rand_acc":       round(rand_acc, 4),
                "selectivity":    0.0,
                **{k: round(v, 4) for k, v in rand_res.items()},
            })
            print(f" Random baseline (слой 12): acc={rand_acc:.3f}")

        # Собираем все группы для анализа: L1 и resource
        all_groups = []
        for l1_name, group_idx in l1_groups.items():
            if check_group_size(y_all[group_idx]):
                all_groups.append((l1_name, group_idx, "n/a"))
        for rt, group_idx in resource_groups.items():
            if check_group_size(y_all[group_idx]):
                all_groups.append((f"resource_{rt}", group_idx, rt))

        if not all_groups:
            print("    Нет подходящих групп — пропускаем")
            continue

        print(f"    Группы для анализа: {[g[0] for g in all_groups]}")

        group_layer_accs = {g[0]: [] for g in all_groups}  # накопитель acc по слоям

        # Цикл по слоям
        for layer_idx in range(n_layers):
            emb_path = emb_dir / f"{emb_prefix}_layer{layer_idx:02d}.npy"
            print(f"    Слой {layer_idx:02d} — загружаю {emb_path.name} ...", end=" ", flush=True)

            if not emb_path.exists():
                print("НЕ НАЙДЕН — пропускаю")
                continue

            X_full = np.load(emb_path)
            X_task = X_full[mask_arr]
            del X_full

            for group_name, group_idx, rt in all_groups:
                X_group = X_task[group_idx]
                y_group = y_all[group_idx]

                res = probe(X_group, y_group)
                selectivity = (res["acc_mean"] - rand_acc) if rand_acc is not None else None
                group_layer_accs[group_name].append(res["acc_mean"])

                all_results.append({
                    "model":          model_alias,
                    "task":           task_name,
                    "level":          level,
                    "l1":             group_name,
                    "resource_type":  rt,
                    "layer":          layer_idx,
                    "n_items":        len(y_group),
                    "majority":       round(majority, 4),
                    "rand_acc":       round(rand_acc, 4) if rand_acc is not None else None,
                    "selectivity":    round(selectivity, 4) if selectivity is not None else None,
                    **{k: round(v, 4) for k, v in res.items()},
                })

            del X_task
            summary = "  ".join(
                f"{g[0]}={group_layer_accs[g[0]][-1]:.3f}"
                for g in all_groups if group_layer_accs[g[0]]
            )
            print(f"готово  [{summary}]")

        # Выводим итоговую точность для L1-групп (слой 12, selectivity)
        for group_name, _, _ in all_groups:
            if group_name.startswith("resource_"):
                continue
            layer_accs = group_layer_accs[group_name]
            if len(layer_accs) == n_layers:
                sel_str = f"  sel={layer_accs[-1] - rand_acc:.3f}" if rand_acc else ""
                print(f"    {group_name}: acc12 = {layer_accs[-1]:.3f}{sel_str}")

        # Промежуточное сохранение
        pd.DataFrame(all_results).to_csv(results_dir / "probing_results_partial.csv", index=False)
        print(f" Промежуточное сохранение: {len(all_results)} строк")

# Финальное сохранение
results_df = pd.DataFrame(all_results)
results_df.to_csv(results_dir / "probing_results.csv", index=False)
print(f"\nРезультаты сохранены: {len(results_df)} строк в {results_dir}/probing_results.csv")

real_l1_df = results_df[
    ~results_df["l1"].isin(["random_baseline"]) &
    ~results_df["l1"].str.startswith("resource_", na=False)
].copy()

print("пробинг закончен")
