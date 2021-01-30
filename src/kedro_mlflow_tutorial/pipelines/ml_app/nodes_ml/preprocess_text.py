import pandas as pd
from string import punctuation
from keras_preprocessing.text import Tokenizer


def lowerize_text(data):
    data.loc[:, "text"] = data.loc[:, "text"].str.lower()
    return data


def remove_stopwords(data, stopwords):
    pattern = r"\b(?:{})\b".format("|".join(stopwords))
    data.loc[:, "text"] = data.loc[:, "text"].str.replace(pattern, "")
    return data


def remove_punctuation(data):
    data.loc[:, "text"] = data.loc[:, "text"].str.replace(
        f"[{punctuation}]", "", regex=True
    )
    return data


def convert_data_to_list(data):
    list_data = list(data.loc[:, "text"])
    return list_data


def fit_tokenizer(list_data, num_words):
    keras_tokenizer = Tokenizer(num_words)
    keras_tokenizer.fit_on_texts(list_data)
    return keras_tokenizer


def tokenize_text(tokenizer, list_data):
    tokenized_text = tokenizer.texts_to_matrix(list_data, mode="tfidf")
    colnames = ["__unused_keras_internal__"] + list(tokenizer.word_index.keys())[
        0 : (tokenizer.num_words - 1)
    ]
    oh_df = pd.DataFrame(data=tokenized_text, columns=colnames)
    oh_df.drop("__unused_keras_internal__", axis=1, inplace=True)
    return oh_df
