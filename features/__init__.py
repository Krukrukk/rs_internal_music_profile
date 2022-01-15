from typing import Any, Union
import json

import torch
from transformers import BertTokenizer, BertModel
import pandas as pd

from logger import get_logger

logger = get_logger(__name__)


class FeatureProcessor:
    """
    Class used to preprocess data

    Attributes
    --------------------------------
    numeric_features : dict[str, dict[str, Any]]
        Json describing names (as a key) and params (min_val, max_val, desc) for features
    text_features : list[str]
        List of strings with names of text features
    section_feature: dict[str, Any]
        Json describing sequence feature from spotify.
        Currently only one of features from sections data can be used
    bert_model : BertModel
        Model used to generate text embeddings
    tokenizer: BertTokenizer
        Tokenizer used by model
    device: str
        cpu/cuda. Device used by torch
    """

    pd.DataFrame()

    def __init__(
        self,
        bert_model: BertModel,
        tokenizer: BertTokenizer,
        device: str = "cuda",
        config_path: str = "features/config/features_config.json",
    ):
        """
        Params
        --------------------------
        model : BertModel
            Model used to generate text embeddings
        tokenizer: BertTokenizer
            Tokenizer used by model
        device: str
            cpu/cuda. Device used by torch
        """
        self.bert_model = bert_model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        (
            self.numeric_features,
            self.text_features,
            self.section_feature,
        ) = FeatureProcessor._load_features(config_path)

    @staticmethod
    def _load_features(config_path: str) -> dict[str, dict]:
        """
        Loads the numeric features config json file
        """
        with open(config_path) as file:
            config = json.load(file)

        return (
            config["numeric_features"],
            config["text_features"],
            config["section_feature"],
        )

    def _preprocess_numeric_features(self, df: pd.DataFrame) -> None:
        """
        Preprocesses numeric using given config. Everything is done inplace

        Params
        --------------------------
        df : pd.DataFrame
            DataFrame with songs data to be preprocessed

        Returns
        ------------------------
        None
        """
        # Process numeric features
        for key in self.numeric_features:
            df[key] = torch.tensor(
                (df[key] - self.numeric_features[key]["min_val"])
                / (
                    self.numeric_features[key]["max_val"]
                    - self.numeric_features[key]["min_val"]
                )
            ).float()

    def _preprocess_text_features(self, df: pd.DataFrame) -> None:
        """
        Preprocesses text using given config. Everything is done inplace

        Params
        --------------------------
        df : pd.DataFrame
            DataFrame with songs data to be preprocessed

        Returns
        ------------------------
        None
        """

        embeddings = [
            (
                self.bert_model(
                    **self.tokenizer(
                        list(row[self.text_features]),
                        return_tensors="pt",
                        padding=True,
                    ).to(self.device)
                )
                .last_hidden_state.mean(dim=1)
                .flatten()
                .detach()
                .float()
                .cpu()
            )
            for _, row in df.iterrows()
        ]
        # logger.info("-----------------------------------------")
        # logger.info(len(df))
        # logger.info(embedding.shape)
        # logger.info("-----------------------------------------")

        df["text_embeddings"] = embeddings

    def _preprocess_section_features(self, df: pd.DataFrame) -> None:
        """
        Preprocesses sections (sequence data from API) using given config. Everything is done inplace

        ParamsEmbedder
        --------------------------
        df : pd.DataFrame
            DataFrame with songs data to be preprocessed

        Returns
        ------------------------
        None
        """
        # Process sections data
        seq_col = f"{self.section_feature['name']}_seq"
        df[seq_col] = df.analysis_sections.apply(
            lambda section: torch.tensor(
                [
                    item[self.section_feature["name"]]
                    for item in (section if section is not None else [])
                ]
            )
        )
        # Normalize/standarize data
        df[seq_col] = df[seq_col].apply(
            lambda row: torch.tensor(
                [
                    (
                        (x - self.section_feature["min_val"])
                        / (
                            self.section_feature["max_val"]
                            - self.section_feature["min_val"]
                        )
                    )
                    * 2
                    - 1
                    for x in row
                ]
            )
        )

        # pad to 14
        df[seq_col] = df[seq_col].apply(lambda x: x[:14])
        df[seq_col] = df[seq_col].apply(
            lambda x: torch.cat(
                (
                    x,
                    torch.tensor([0] * (14 - len(x))),
                ),
                dim=0,
            )
            .reshape(-1, 1)
            .float()
        )

    def preprocess_features(self, df: pd.DataFrame) -> None:
        """
        Get preprocessed features. Everything is done in place

        df : pd.DataFrame
            DataFrame with songs data to be preprocessed
        """
        self._preprocess_numeric_features(df)
        self._preprocess_text_features(df)
        self._preprocess_section_features(df)
