# -*- coding: utf-8 -*-
"""
probing_by_level.py
Пробинг по уровню владения русским языком (CEFR: A1–C1).

Методология полностью совпадает с probing.py, но группы определяются
не по родному языку (L1), а по уровню владения.

Два режима группировки (оба запускаются в одном скрипте):

  РЕЖИМ А — группы внутри каждого L1 (per-L1):
      ключ группы = "{dominant_language}_{level_of_rus}"  например "english_B2"
      Позволяет сравнить B2 и C1 внутри одного и того же L1.
      Результаты сохраняются с префиксом "bylevel_perlang".

  РЕЖИМ Б — объединённые группы (pooled):
      ключ группы = "{level_of_rus}"  например "B2"
      Все L1 слиты, только уровни.
      Позволяет увидеть общую картину: меняется ли кодирование морфологии/синтаксиса
      с ростом уровня владения языком.
      Результаты сохраняются с префиксом "bylevel_pooled".

Выходные файлы
  results/probing_bylevel_perlang.csv   (режим A)
  results/probing_bylevel_pooled.csv    (режим Б)

Примечания
  - Чешский и болгарский не имеют текстов уровней A1/A2/B1 (см. 02_extract_embeddings.py).
    В тепловых картах эти ячейки будут NaN.
  - Казахский — на 100% эритажные носители, поэтому группировка по уровням
    может смешивать уровень L2 с особенностями наследия; интерпретировать осторожно.
  - MIN_SAMPLES_PER_CLASS такой же, как в 03_probing.py (20). Группы с меньшим
    количеством примеров молча пропускаются.
  - Оптимизация загрузки слоёв: каждый .npy загружается один раз на задачу,
    потом режется по всем группам — как в 03_probing.py v2.
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

# ========== ПУТИ ==========
emb_dir = Path("embeddings")
results_dir = Path("results")
plots_dir = results_dir / "plots" / "bylevel"
results_dir.mkdir(exist_ok=True)
plots_dir.mkdir(parents=True, exist_ok=True)

# параметры должны совпадать с probing.py
models_list = ["xlmr"]           # какие модели обрабатываем
n_layers = 13                    # слои 0..12
n_folds = 3                      # фолды кросс-валидации
min_samples_per_class = 20       # минимальное число примеров в классе
lr_params = {"C": 1.0, "max_iter": 2000, "random_state": 42}

# Порядок уровней CEFR (от низкого к высокому)
level_order = ["A1", "A2", "B1", "B2", "C1"]


# Здесь только синтаксические задачи (как в оригинале)
tasks = { # для демонстрации, что код работает, закоммитила все задачи, кроме Tense, потому что он выполняется относительно быстрее остальных
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



# ========== ЦВЕТА ==========
# Цветовая палитра для уровней: от холодного (A1) к тёплому (C1)
level_palette = {
    "A1": "#90CAF9",   # светло-голубой
    "A2": "#42A5F5",   # синий
    "B1": "#FFA726",   # оранжевый
    "B2": "#EF5350",   # красный
    "C1": "#B71C1C",   # тёмно-красный
}

# Порядок отображения L1 (как в 03_probing.py)
l1_display_order = [
    "arabic", "chinese", "english", "french", "german",
    "italian", "japonese", "kazah", "korean", "swedish",
    "bulgarian", "finnish", "hindi", "hungarian",
    "norwegian", "pushtu", "thai", "\u0441zech",
]


def ordered_levels(available: list) -> list:
    """Сортирует уровни по порядку CEFR; неизвестные дописывает в конец."""
    seen, result = set(), []
    for lv in level_order:
        if lv in available and lv not in seen:
            result.append(lv); seen.add(lv)
    for lv in available:
        if lv not in seen:
            result.append(lv); seen.add(lv)
    return result


def ordered_l1s(available: list) -> list:
    seen, result = set(), []
    for l1 in l1_display_order:
        if l1 in available and l1 not in seen:
            result.append(l1); seen.add(l1)
    for l1 in available:
        if l1 not in seen:
            result.append(l1); seen.add(l1)
    return result

# функции пробинга как в probing.py

def probe(X: np.ndarray, y: np.ndarray) -> dict:
    """Обучает логистическую регрессию с кросс-валидацией."""
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
    """Пробинг на случайных эмбеддингах (контроль)."""
    X_rand = np.random.default_rng(42).standard_normal(X.shape).astype(np.float32)
    return probe(X_rand, y)

def majority_baseline(y: np.ndarray) -> float:
    """Доля самого частотного класса."""
    _, counts = np.unique(y, return_counts=True)
    return float(counts.max() / len(y))

def check_group_size(y: np.ndarray) -> bool:
    """Проверяет, что в группе есть хотя бы два класса и достаточно примеров на класс."""
    _, counts = np.unique(y, return_counts=True)
    return len(counts) >= 2 and int(counts.min()) >= min_samples_per_class


def run_probing(group_defs: list, y_all: np.ndarray, mask_arr: np.ndarray,
                emb_prefix: str, rand_acc: float | None,
                model_alias: str, task_name: str, level: str,
                majority: float) -> list:
    """
    group_defs: список кортежей (group_key, row_indices, meta_dict)
      group_key   — строка-идентификатор группы (будет в колонке 'group')
      row_indices — индексы строк в task_index (и в y_all)
      meta_dict   — дополнительные колонки для результата (l1, prof_level, mode)

    Возвращает список словарей с результатами (по одной строке на группу на слой).
    """
    if not group_defs:
        return []

    results = []
    # Для каждого слоя будем накапливать точности, чтобы потом не надо (но дельты не нужны)
    # Однако нам нужно только собрать результаты, дельты не считаем.

    for layer_idx in range(n_layers):
        emb_path = emb_dir / f"{emb_prefix}_layer{layer_idx:02d}.npy"
        print(f"    Слой {layer_idx:02d} — загружаю {emb_path.name} ...", end=" ", flush=True)

        if not emb_path.exists():
            print("НЕ НАЙДЕН — пропускаю")
            continue

        X_full = np.load(emb_path)
        X_task = X_full[mask_arr]
        del X_full

        layer_summary = []
        for group_key, group_idx, meta in group_defs:
            y_group = y_all[group_idx]
            if not check_group_size(y_group):
                continue

            X_group = X_task[group_idx]
            res = probe(X_group, y_group)
            selectivity = (res["acc_mean"] - rand_acc) if rand_acc is not None else None
            layer_summary.append(f"{group_key}={res['acc_mean']:.3f}")

            results.append({
                "model":          model_alias,
                "task":           task_name,
                "task_level":     level,            # 'token' или 'sent'
                "layer":          layer_idx,
                "n_items":        len(y_group),
                "majority":       round(majority, 4),
                "rand_acc":       round(rand_acc, 4) if rand_acc is not None else None,
                "selectivity":    round(selectivity, 4) if selectivity is not None else None,
                **{k: round(v, 4) for k, v in res.items()},  # acc_mean, acc_std, f1_mean, f1_std
                **meta,           # group, l1, prof_level, mode
            })

        del X_task
        print(f"готово  [{' | '.join(layer_summary)}]")

    return results



results_perlang = []   # режим A: внутри каждого L1 по уровням
results_pooled  = []   # режим Б: все L1 слиты, только уровни

for model_alias in models_list:
    print(f"\nМОДЕЛЬ: {model_alias}")

    # Загружаем индексные файлы
    index_token = pd.read_csv(emb_dir / f"index_{model_alias}.csv")
    index_sent  = pd.read_csv(emb_dir / f"index_sent_{model_alias}.csv")
    index_token.columns = index_token.columns.str.lower()
    index_sent.columns  = index_sent.columns.str.lower()

    for task_name, task_cfg in tasks.items():
        level = task_cfg["level"]    # 'token' или 'sent'
        col   = task_cfg["col"]
        upos  = task_cfg["upos"]

        print(f"\n  Задача: {task_name}  (уровень={level})")

        index = index_token if level == "token" else index_sent
        emb_prefix = model_alias if level == "token" else f"sent_{model_alias}"

        # Маска: не пропущен признак И есть уровень владения русским
        mask = index[col].notna() & index["level_of_rus"].notna()
        if upos is not None and "upos" in index.columns:
            mask = mask & (index["upos"].isin(upos))

        task_index = index[mask].reset_index(drop=True)
        mask_arr = mask.values

        if len(task_index) < min_samples_per_class * 2:
            print(f"    Недостаточно данных ({len(task_index)}) — пропускаем")
            continue

        # Кодируем метки целевого признака в числа
        le = LabelEncoder()
        y_all = le.fit_transform(task_index[col].values)
        print(f"    Классы: {list(le.classes_)}  |  примеров: {len(task_index):,}")

        majority = majority_baseline(y_all)
        print(f"    Majority baseline: {majority:.3f}")

        # Random baseline (слой 12)
        rand_acc = None
        last_path = emb_dir / f"{emb_prefix}_layer12.npy"
        if last_path.exists():
            X_last = np.load(last_path)[mask_arr]
            rand_res = probe_random(X_last, y_all)
            rand_acc = rand_res["acc_mean"]
            del X_last
            print(f"    Random baseline (слой 12): acc={rand_acc:.3f}")

        # ----- РЕЖИМ А: группы внутри каждого L1 (L1_уровень) -----
        print("\n  [Режим А] Группы по L1 и уровню (per-L1)")
        perlang_defs = []
        for (l1, lv), grp in task_index.groupby(["dominant_language", "level_of_rus"]):
            group_key = f"{l1}_{lv}"
            y_group = y_all[grp.index.values]
            if check_group_size(y_group):
                perlang_defs.append((
                    group_key,
                    grp.index.values,
                    {"group": group_key, "l1": l1, "prof_level": lv, "mode": "per_lang"},
                ))

        if perlang_defs:
            rows_a = run_probing(
                perlang_defs, y_all, mask_arr, emb_prefix, rand_acc,
                model_alias, task_name, level, majority,
            )
            # Добавляем строку для random baseline (как отдельную "группу")
            if rand_acc is not None:
                rows_a.append({
                    "model": model_alias, "task": task_name, "task_level": level,
                    "layer": 12, "n_items": len(y_all),
                    "majority": round(majority, 4), "rand_acc": round(rand_acc, 4),
                    "selectivity": 0.0,
                    "acc_mean": round(rand_res["acc_mean"], 4),
                    "acc_std": round(rand_res["acc_std"], 4),
                    "f1_mean": round(rand_res["f1_mean"], 4),
                    "f1_std": round(rand_res["f1_std"], 4),
                    "group": "random_baseline", "l1": "random_baseline",
                    "prof_level": "n/a", "mode": "per_lang",
                })
            results_perlang.extend(rows_a)

        # ----- РЕЖИМ Б: объединённые группы (только уровень) -----
        print("\n  [Режим Б] Объединённые группы по уровню (pooled)")
        pooled_defs = []
        for lv, grp in task_index.groupby("level_of_rus"):
            y_group = y_all[grp.index.values]
            if check_group_size(y_group):
                pooled_defs.append((
                    lv,
                    grp.index.values,
                    {"group": lv, "l1": "all", "prof_level": lv, "mode": "pooled"},
                ))

        if pooled_defs:
            rows_b = run_probing(
                pooled_defs, y_all, mask_arr, emb_prefix, rand_acc,
                model_alias, task_name, level, majority,
            )
            if rand_acc is not None:
                rows_b.append({
                    "model": model_alias, "task": task_name, "task_level": level,
                    "layer": 12, "n_items": len(y_all),
                    "majority": round(majority, 4), "rand_acc": round(rand_acc, 4),
                    "selectivity": 0.0,
                    "acc_mean": round(rand_res["acc_mean"], 4),
                    "acc_std": round(rand_res["acc_std"], 4),
                    "f1_mean": round(rand_res["f1_mean"], 4),
                    "f1_std": round(rand_res["f1_std"], 4),
                    "group": "random_baseline", "l1": "all",
                    "prof_level": "n/a", "mode": "pooled",
                })
            results_pooled.extend(rows_b)

        # Промежуточное сохранение
        pd.DataFrame(results_perlang).to_csv(
            results_dir / "probing_bylevel_perlang_partial.csv", index=False)
        pd.DataFrame(results_pooled).to_csv(
            results_dir / "probing_bylevel_pooled_partial.csv", index=False)
        print(f"\n    Промежуточное сохранение: {len(results_perlang)} строк (режим А), "
              f"{len(results_pooled)} строк (режим Б)")


df_perlang = pd.DataFrame(results_perlang)
df_pooled  = pd.DataFrame(results_pooled)
df_perlang.to_csv(results_dir / "probing_bylevel_perlang.csv", index=False)
df_pooled.to_csv(results_dir / "probing_bylevel_pooled.csv", index=False)

print(f"\nГотово!!")
print(f"Сохранено {len(df_perlang)} строк (режим А) сохранено в  probing_bylevel_perlang.csv")
print(f"Сохранено {len(df_pooled)} строк (режим Б) срхранено в probing_bylevel_pooled.csv")
