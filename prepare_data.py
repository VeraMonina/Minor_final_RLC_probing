# -*- coding: utf-8 -*-
"""
prepare_data.py

Загружаем датасет RLC, делаем токенизацию, морфологическую разметку через Stanza,
извлекаем синтаксические признаки на уровне предложения.

На выходе два файла:
    - annotated_tokens.csv  — пословная разметка (морфология + позиция в предложении)
    - annotated_sents.csv   — по предложениям (длина, порядок слов, позиция подлежащего и сказуемого)

Морфологические категории, которые мы отслеживаем:
    - Case     (падеж) для сущ., местоим., прил., детерминативов, числит.
    - Number   (число) для сущ., местоим., прил., детерм., числит., глаголов
    - Aspect   (вид) для глаголов и вспом. глаголов
    - Tense    (время) для глаголов и вспом.
    - Gender   (род) для сущ., прил., детерм., глаголов
    - Person   (лицо) для глаголов, вспом., местоим.
    - Mood     (наклонение) для глаголов и вспом.

Синтаксические категории:
    - sent_len       — длина предложения в токенах (с разбивкой на короткие/средние/длинные)
    - subj_position  — позиция подлежащего относительно сказуемого: first / non-first
    - verb_position  — позиция сказуемого в предложении: early / middle / late
    - svo_order      — порядок слов: SVO, SOV, VSO, VOS, OVS, OSV, no_obj
"""

from pathlib import Path
import pandas as pd
import stanza

# задаем нужные нам константы
length_bins = [0, 7, 15, 9999]
length_labels = ["short", "medium", "long"]

# Морфологические признаки, которые будем вытаскивать
target_features = {
    "Case",
    "Number",
    "Aspect",
    "Tense",
    "Gender",
    "Person",
    "Mood",
}

# Для каждого признака — какие части речи имеют смысл (фильтр)
upos_feature_filter = {
    "Case": {"NOUN", "PRON", "ADJ", "DET", "NUM"},
    "Number": {"NOUN", "PRON", "ADJ", "DET", "NUM", "VERB"},
    "Gender": {"NOUN", "ADJ", "DET", "VERB"},
    "Aspect": {"VERB", "AUX"},
    "Tense": {"VERB", "AUX"},
    "Person": {"VERB", "AUX", "PRON"},
    "Mood": {"VERB", "AUX"},
}


df = pd.read_csv("dataset_RLC_final_project.csv")
texts = df.to_dict("records")   # список словарей, удобно для итерации


# Скачиваем модель
stanza.download("ru", verbose=False)

# Создаём пайплайн: токенизация, разбор по частям речи, лемматизация, синтаксические зависимости
nlp = stanza.Pipeline(
    lang="ru",
    processors="tokenize,pos,lemma,depparse",
    use_gpu=False,     
    verbose=False,        # чтобы не выводил кучу служебных сообщений
)


# вспомогательные функции для удобной разметки
def parse_features(features_string):
    """
    Превращает строку типа "Case=Nom|Number=Sing" в словарь {'Case': 'Nom', 'Number': 'Sing'}.
    Если строка пустая или '_' — возвращаем пустой словарь.
    """
    if not features_string or features_string == "_":
        return {}
    return dict(pair.split("=") for pair in features_string.split("|"))

def get_sentence_length_category(token_count):
    """
    По числу токенов возвращает категорию длины предложения:
        short   — до 7 токенов
        medium  — 8–15 токенов
        long    — 16 и больше
    """
    if token_count <= 7:
        return "short"
    elif token_count <= 15:
        return "medium"
    return "long"

def get_word_order(sentence):
    """
    Определяет порядок слов (SVO, SOV, ...), позицию подлежащего и сказуемого.
    Всё на основе дерева зависимостей (Universal Dependencies).

    Возвращает три строки:
        word_order       — например 'SVO', 'SOV', 'no_subj', 'no_obj'
        subject_position — 'first' (подлежащее раньше сказуемого) или 'non-first'
        verb_position    — 'early', 'middle', 'late'
    """
    words = sentence.words
    sentence_length = len(words)

    root_index = None      # индекс сказуемого (корень дерева)
    subject_index = None   # индекс подлежащего
    object_index = None    # индекс прямого дополнения

    # Проходим по всем словам в предложении
    for word in words:
        if word.deprel == "root":
            root_index = word.id - 1   # нумерация с 1, переводим в 0-based
        elif word.deprel in {"nsubj", "nsubj:pass"} and subject_index is None:
            subject_index = word.id - 1
        elif word.deprel in {"obj", "iobj"} and object_index is None:
            object_index = word.id - 1

    # позиция сказуемого (verb_position)
    if root_index is None:
        verb_position = "unknown"
    elif root_index < sentence_length / 3:
        verb_position = "early"
    elif root_index < 2 * sentence_length / 3:
        verb_position = "middle"
    else:
        verb_position = "late"

    # позиция подлежащего относительно сказуемого (subject_position)
    if subject_index is None or root_index is None:
        subject_position = "no_subj"
    elif subject_index < root_index:
        subject_position = "first"
    else:
        subject_position = "non-first"

    # порядок слов S/V/O 
    if root_index is None:
        word_order = "unknown"
    elif subject_index is None:
        word_order = "no_subj"
    elif object_index is None:
        word_order = "no_obj"
    else:
        # Сортируем тройку (S, V, O) по их позициям в предложении
        order = sorted(
            [
                ("S", subject_index),
                ("V", root_index),
                ("O", object_index),
            ],
            key=lambda item: item[1]   # по индексу
        )
        word_order = "".join(role for role, _ in order)

    return word_order, subject_position, verb_position



token_data = []      # сюда будем складывать токены
sentence_data = []   # сюда — информацию по предложениям
save_every = 100     # каждые 100 текстов сохраняем промежуточные чекпоинты, потому что разметка занимает много времени

print("Начинаем обработку текстов...")

for text_idx, text_row in enumerate(texts):
    # Прогоняем текст через пайплайн Stanza
    doc = nlp(text_row["text"])

    # Обрабатываем каждое предложение внутри документа
    for sent_id, sentence in enumerate(doc.sentences):
        words = sentence.words
        token_count = len(words)

        # Получаем синтаксические характеристики предложения
        word_order, subj_pos, verb_pos = get_word_order(sentence)

        # Сохраняем информацию о предложении
        sentence_data.append({
            "text_id": text_row["text_id"],
            "sent_id": sent_id,
            "sent_len": token_count,
            "sent_len_bin": get_sentence_length_category(token_count),
            "svo_order": word_order,
            "subj_position": subj_pos,
            "verb_position": verb_pos,
            "dominant_language": text_row["dominant_language"],
            "level_of_rus": text_row["level_of_rus"],
            "background": text_row["background"],
        })

        # Обрабатываем токены внутри предложения
        for token_id, word in enumerate(words):
            # Извлекаем морфологические признаки из поля feats
            feats_dict = parse_features(word.feats)

            # Оставляем только те признаки, которые нас интересуют,
            # и только для разрешённых частей речи
            relevant = {}
            for feat in target_features:
                if feat in feats_dict and word.upos in upos_feature_filter.get(feat, set()):
                    relevant[feat] = feats_dict[feat]

            # Если ни одного значимого признака нет — пропускаем этот токен
            if not relevant:
                continue

            # Определяем позицию токена в предложении (начало/середина/конец)
            rel_pos = token_id / max(token_count - 1, 1)   # относительная позиция от 0 до 1
            if rel_pos < 0.33:
                pos_cat = "beginning"
            elif rel_pos < 0.67:
                pos_cat = "middle"
            else:
                pos_cat = "end"

            # Собираем запись для токена
            token_data.append({
                "text_id": text_row["text_id"],
                "sent_id": sent_id,
                "token_id": token_id,
                "word": word.text,
                "lemma": word.lemma,
                "upos": word.upos,
                "deprel": word.deprel,
                "Case": feats_dict.get("Case"),
                "Number": feats_dict.get("Number"),
                "Aspect": feats_dict.get("Aspect"),
                "Tense": feats_dict.get("Tense"),
                "Gender": feats_dict.get("Gender"),
                "Person": feats_dict.get("Person"),
                "Mood": feats_dict.get("Mood"),
                "position_bin": pos_cat,
                "sent_len_bin": get_sentence_length_category(token_count),
                "dominant_language": text_row["dominant_language"],
                "level_of_rus": text_row["level_of_rus"],
                "background": text_row["background"],
            })

    # Раз в save_every текстов сохраняем чекпоинты, чтобы не потерять прогресс
    if (text_idx + 1) % save_every == 0:
        token_checkpoint = f"tokens_checkpoint_{text_idx + 1}.csv"
        sent_checkpoint = f"sentences_checkpoint_{text_idx + 1}.csv"
        pd.DataFrame(token_data).to_csv(token_checkpoint, index=False, encoding="utf-8")
        pd.DataFrame(sentence_data).to_csv(sent_checkpoint, index=False, encoding="utf-8")
        print(f"  Сохранён чекпоинт после {text_idx + 1} текстов")


tokens_df = pd.DataFrame(token_data)
sentences_df = pd.DataFrame(sentence_data)

tokens_df.to_csv("annotated_tokens.csv", index=False, encoding="utf-8")
sentences_df.to_csv("annotated_sents.csv", index=False, encoding="utf-8")

print("сохранены ура annotated_tokens.csv и annotated_sents.csv")
