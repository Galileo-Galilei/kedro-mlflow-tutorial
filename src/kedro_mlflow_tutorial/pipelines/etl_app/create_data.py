import pandas as pd
from datasets import (
    load_dataset,
)  # see: https://huggingface.co/nlp/viewer/?dataset=imdb

from typing import Union


def create_instances(split: str) -> Union[pd.DataFrame, pd.DataFrame]:
    """This functions emulates areal etl by creating instances 
    and labels as if they were quried from a database

    Args:
        split ([str]): Filter the dataset depending on the original imdb
        split. Either "train", "test" or "unsupervised"
    """

    huggingface_dataset = load_dataset("imdb", split=split)

    huggingface_dataset.set_format(type="pandas", columns="text")
    instances = huggingface_dataset[:]

    return instances


def create_labels(split: str) -> Union[pd.DataFrame, pd.DataFrame]:
    """This functions emulates areal etl by creating instances 
    and labels as if they were quried from a database

    Args:
        split ([str]): Filter the dataset depending on the original imdb
        split. Either "train", "test" or "unsupervised"
    """

    huggingface_dataset = load_dataset("imdb", split=split)

    huggingface_dataset.set_format(type="pandas", columns="label")
    labels = huggingface_dataset[:]

    # simulate the fact that labels are not encoded yet
    # this would be more realistic in case of scraping
    labels.loc[:, "label"].replace(
        to_replace={0: "negative", 1: "positive"}, inplace=True
    )

    return labels

