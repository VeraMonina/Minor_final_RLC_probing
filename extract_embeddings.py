# -*- coding: utf-8 -*-
"""
extract_embeddings.py

Извлекаем эмбеддинги на уровне токенов и предложений для трёх моделей:
  - rubert  → DeepPavlov/rubert-base-cased
  - mbert   → google-bert/bert-base-multilingual-cased
  - xlmr    → FacebookAI/xlm-roberta-base

Для каждой модели сохраняем два типа эмбеддингов:
  - токеновые (усреднённые по субтокенам) → {model}_layer{N}.npy
  - по предложениям (усреднённые по всем токенам, без CLS/SEP) → sent_{model}_layer{N}.npy

А также индексные файлы (CSV) с метаданными, чтобы знать, какому токену/предложению
соответствует каждая строка в .npy.

Важное замечание по памяти: раньше я пытался накопить ВСЕ векторы в RAM, но на ~21k
предложений вылезала ошибка out-of-memory. Поэтому теперь я сбрасываю буферы на диск
каждые flush_every предложений во временные чанки, а в конце склеиваю их в финальные
.npy файлы. Так память не взрывается.
"""

import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

num_layers = 13         # слои: 0 = эмбеддинговый, 1..12 = трансформерные блоки
flush_every = 500       # сбрасываем на диск каждые 500 предложений (меньше = меньше RAM, но больше файлов)

tokens_path = "annotated_tokens.csv"
sents_path  = "annotated_sents.csv"
emb_dir     = "embeddings"
models_dir  = "models"

os.makedirs(emb_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Это списки языков, которые мы считаем высокоресурсными (>50 текстов)
high_resource = ["arabic", "chinese", "english", "french", "german",
                 "italian", "japonese", "kazah", "korean", "swedish"]

low_resource  = ["bulgarian", "finnish", "hindi", "hungarian",
                 "norwegian", "pushtu", "thai", "сzech"]

print("High-resource L1s:", high_resource)
print("Low-resource  L1s:", low_resource)

# Словарь: имя модели → путь к ней (или имя на HuggingFace)
models = {
    "rubert": "DeepPavlov/rubert-base-cased",
    "mbert":  "google-bert/bert-base-multilingual-cased",
    "xlmr":   "xlm-roberta-base",
}


print(f"Всего моделей для обработки: {len(models)}")


df_tokens = pd.read_csv(tokens_path)
df_sents  = pd.read_csv(sents_path)
print(f"  Токенов:    {len(df_tokens):,}")
print(f"  Предложений: {len(df_sents):,}")

# Группируем токены по (text_id, sent_id), чтобы потом итерировать по предложениям
# Превращаем GroupBy в список, чтобы можно было проходиться несколько раз (если понадобится)
sent_groups = list(df_tokens.groupby(["text_id", "sent_id"]))
print(f"  Уникальных предложений: {len(sent_groups):,}")

# Делаем быстрый индекс для df_sents: словарь (text_id, sent_id) → строка метаданных
# Без этого внутри цикла приходилось бы каждый раз искать по всему датафрейму (очень медленно)
sent_index = df_sents.set_index(["text_id", "sent_id"]).sort_index()


def get_subtoken_spans(tokenizer, words):
    """
    Токенизируем список слов (уже разбитых) через BPE/WordPiece.
    Возвращаем:
      - encoding: входные id, attention_mask и т.д. (для подачи в модель)
      - spans: словарь {индекс_слова: (начало_позиции, конец_позиции)} в последовательности субтокенов.
    Это нужно, потому что одно слово может разбиться на несколько субтокенов,
    и чтобы получить эмбеддинг слова, мы усредним все субтокены, которые ему принадлежат.
    """
    encoding = tokenizer(
        words,
        is_split_into_words=True,      # слова уже предварительно разбиты пробелами
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    word_ids = encoding.word_ids()      # для каждой позиции субтокена — номер слова (или None для CLS/SEP)
    spans = {}
    for pos, wid in enumerate(word_ids):
        if wid is None:
            continue
        if wid not in spans:
            spans[wid] = [pos, pos + 1]   # [start, end) — полуинтервал
        else:
            spans[wid][1] = pos + 1
    return encoding, {k: tuple(v) for k, v in spans.items()}

def pool_word(hidden_states, span, layer_idx):
    """
    Усредняет субтокены одного слова на заданном слое.
    hidden_states: tuple из num_layers тензоров (1, seq_len, hidden_size)
    """
    layer = hidden_states[layer_idx]          # (1, seq_len, hidden_size)
    vecs = layer[0, span[0]:span[1], :]       # вырезаем позиции субтокенов
    return vecs.mean(dim=0).cpu().numpy()      # усредняем и переводим в numpy

def pool_sentence(hidden_states, layer_idx, encoding):
    """
    Усредняет все содержательные токены (исключая [CLS] и [SEP]) для получения
    эмбеддинга предложения. Это стандартная практика для базовых BERT-моделей,
    которые не дообучались на классификацию: [CLS] в них не оптимизирован.
    """
    layer = hidden_states[layer_idx]          # (1, seq_len, hidden_size)
    # Длина последовательности без учёта спецтокенов: attention_mask суммирует все реальные токены,
    # вычитаем 2 (CLS и SEP) — но только если они есть; на всякий случай берём max(1)
    actual_len = int(encoding["attention_mask"].sum()) - 2
    actual_len = max(actual_len, 1)
    vecs = layer[0, 1:actual_len + 1, :]       # берём токены от позиции 1 до actual_len
    return vecs.mean(dim=0).cpu().numpy()

def flush_chunk(token_bufs, sent_bufs, model_name, chunk_idx, emb_dir, num_layers):
    """
    Сбрасывает текущие буферы в памяти на диск.
    Каждый слой сохраняется в отдельный файл вида:
        {model_name}_layer{NN}_chunk{KKKK}.npy
        sent_{model_name}_layer{NN}_chunk{KKKK}.npy
    После этого буферы очищаются и возвращаются пустыми.
    """
    for layer in range(num_layers):
        if token_bufs[layer]:
            path = os.path.join(emb_dir, f"{model_name}_layer{layer:02d}_chunk{chunk_idx:04d}.npy")
            np.save(path, np.array(token_bufs[layer], dtype=np.float32))
        if sent_bufs[layer]:
            path = os.path.join(emb_dir, f"sent_{model_name}_layer{layer:02d}_chunk{chunk_idx:04d}.npy")
            np.save(path, np.array(sent_bufs[layer], dtype=np.float32))
    # Возвращаем свежие пустые списки для каждого слоя
    return [[] for _ in range(num_layers)], [[] for _ in range(num_layers)]

def merge_chunks(model_name, emb_dir, num_layers, total_chunks):
    """
    функция клеивает все файлы для одной модели в финальные .npy файлы по слоям.
    """
    print(f"\n  склеивание для модели {model_name}...")
    for prefix in [model_name, f"sent_{model_name}"]:
        for layer in range(num_layers):
            chunks_data = []
            for chunk_idx in range(total_chunks):
                chunk_path = os.path.join(emb_dir, f"{prefix}_layer{layer:02d}_chunk{chunk_idx:04d}.npy")
                if os.path.exists(chunk_path):
                    chunks_data.append(np.load(chunk_path))
            if not chunks_data:
                continue
            merged = np.concatenate(chunks_data, axis=0)
            final_path = os.path.join(emb_dir, f"{prefix}_layer{layer:02d}.npy")
            np.save(final_path, merged)
            print(f"    {prefix}_layer{layer:02d}: {merged.shape}")
            # Удаляем чанки
            for chunk_idx in range(total_chunks):
                chunk_path = os.path.join(emb_dir, f"{prefix}_layer{layer:02d}_chunk{chunk_idx:04d}.npy")
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)


for model_name, model_path in models.items():
    print(f"\nМОДЕЛЬ: {model_name}")

    # Если финальный файл последнего слоя уже существует — пропускаем (возобновление)
    check_file = os.path.join(emb_dir, f"{model_name}_layer12.npy")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, output_hidden_states=True)
    model.eval()
    model.to(device)

    #каждый слой — отдельный список векторов
    token_bufs = [[] for _ in range(num_layers)]
    sent_bufs  = [[] for _ in range(num_layers)]

    # Метаданные (CSV) — накапливаются полностью в памяти
    token_rows = []
    sent_rows  = []

    n_truncated = 0   # счётчик токенов, потерянных из-за обрезания по max_length=512
    chunk_idx = 0

    for i, ((text_id, sent_id), group) in enumerate(sent_groups):
        # group — это DataFrame с токенами внутри одного предложения
        words = group["word"].tolist()

        encoding, spans = get_subtoken_spans(tokenizer, words)
        # Считаем потерянные токены: если слово не попало в spans, значит оно обрезано
        n_truncated += len(words) - len(spans)

        # Прогоняем через модель
        inputs = {k: v.to(device) for k, v in encoding.items()}
        with torch.no_grad():
            outputs = model(**inputs)

        hidden_states = outputs.hidden_states   # tuple из num_layers тензоров

        # Токеновые эмбеддинги 
        for local_idx, row in enumerate(group.itertuples()):
            if local_idx not in spans:
                continue   # слово было обрезано — пропускаем
            span = spans[local_idx]
            for layer in range(num_layers):
                vec = pool_word(hidden_states, span, layer)
                token_bufs[layer].append(vec)

            # Метаданные для этого токена
            resource = "high" if row.dominant_language in high_resource else "low"
            token_rows.append({
                "text_id":           row.text_id,
                "sent_id":           row.sent_id,
                "token_id":          row.token_id,
                "word":              row.word,
                "upos":              row.upos,
                "deprel":            row.deprel,
                "case":              row.Case,
                "number":            row.Number,
                "aspect":            row.Aspect,
                "tense":             row.Tense,
                "gender":            row.Gender,
                "person":            row.Person,
                "mood":              row.Mood,
                "position_bin":      row.position_bin,
                "sent_len_bin":      row.sent_len_bin,
                "dominant_language": row.dominant_language,
                "level_of_rus":      row.level_of_rus,
                "background":        row.background,
                "resource_type":     resource,
            })

        # Эмбеддинги предложений 
        # Быстрый поиск метаданных по индексу (text_id, sent_id)
        try:
            meta = sent_index.loc[(text_id, sent_id)]
        except KeyError:
            # Такое предложение отсутствует в df_sents (например, слишком короткое)
            # Очищаем память и идём дальше
            del hidden_states, outputs, inputs
            if device == "cuda":
                torch.cuda.empty_cache()
            continue

        if isinstance(meta, pd.DataFrame):
            meta = meta.iloc[0]

        resource = "high" if meta["dominant_language"] in high_resource else "low"
        for layer in range(num_layers):
            vec = pool_sentence(hidden_states, layer, encoding)
            sent_bufs[layer].append(vec)
          
        # размечаем данные для эмюелингов по предложениям 
        sent_rows.append({
            "text_id":           text_id,
            "sent_id":           sent_id,
            "sent_len":          meta["sent_len"],
            "sent_len_bin":      meta["sent_len_bin"],
            "svo_order":         meta["svo_order"],
            "subj_position":     meta["subj_position"],
            "verb_position":     meta["verb_position"],
            "dominant_language": meta["dominant_language"],
            "level_of_rus":      meta["level_of_rus"],
            "background":        meta["background"],
            "resource_type":     resource,
        })

        # Очищаем память от тензоров этого предложения (чтобы не копились)
        del hidden_states, outputs, inputs
        if device == "cuda":
            torch.cuda.empty_cache()

        # сброс памяти
        if (i + 1) % flush_every == 0:
            token_bufs, sent_bufs = flush_chunk(token_bufs, sent_bufs, model_name,
                                                chunk_idx, emb_dir, num_layers)
            chunk_idx += 1
            print(f"  Предложений обработано: {i+1:6d} | "
                  f"Токенов сохранено: {len(token_rows):7d} | "
                  f"Чанков сброшено: {chunk_idx} | "
                  f"Обрезано токенов: {n_truncated:4d}")

    # после полного цикла опять сбрасываем
    if any(token_bufs[l] for l in range(num_layers)):
        token_bufs, sent_bufs = flush_chunk(token_bufs, sent_bufs, model_name,
                                            chunk_idx, emb_dir, num_layers)
        chunk_idx += 1

    total_chunks = chunk_idx  

    # Склеиваем все файлы .npy
    merge_chunks(model_name, emb_dir, num_layers, total_chunks)

    # Сохраняем индексные CSV
    pd.DataFrame(token_rows).to_csv(
        os.path.join(emb_dir, f"index_{model_name}.csv"), index=False
    )
    pd.DataFrame(sent_rows).to_csv(
        os.path.join(emb_dir, f"index_sent_{model_name}.csv"), index=False
    )

    print(f"\n  Итог: токенов = {len(token_rows):,}, предложений = {len(sent_rows):,}")
    print(f"  Обрезано токенов (из-за 512): {n_truncated:,}  ← особенно важно для финского")

    # Очищаем модель перед следующей
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\ndone ура")
print(f"Результаты в папке: {emb_dir}")
