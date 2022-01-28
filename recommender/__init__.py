import numpy as np
import pandas as pd
import pickle
import os


class Recommender:
    def __init__(
        self,
    ):
        with open(
            os.path.join("recommender", "knn", "knn_classifier.pkl"), "rb"
        ) as file:
            self.knn_clf = pickle.load(file)
        with open(os.path.join("data", "clusters_tracks.pkl"), "rb") as file:
            self.clusters_tracks = pickle.load(file)

    @staticmethod
    def _occurances2probs_proportional(occurances_count):
        tmp_sum = sum(occurances_count)
        return [i / tmp_sum for i in occurances_count]

    @staticmethod
    def _occurances2probs_square(occurances_count):
        _sqr = np.square(occurances_count)
        return _sqr / np.sum(_sqr)

    @staticmethod
    def _occurances2probs_sqrt(occurances_count):
        _sqrt = np.sqrt(occurances_count)
        return _sqrt / np.sum(_sqrt)

    @staticmethod
    def _occurances2probs_tanh(occurances_count):
        _tanh = np.tanh(occurances_count)
        return _tanh / np.sum(_tanh)

    @staticmethod
    def _occurances2probs_strategy(occurances_count, method):
        if method == "proportional":
            probas = Recommender._occurances2probs_proportional(occurances_count)
        elif method == "square":
            probas = Recommender._occurances2probs_square(occurances_count)
        elif method == "sqrt":
            probas = Recommender._occurances2probs_sqrt(occurances_count)
        elif method == "tanh":
            probas = Recommender._occurances2probs_tanh(occurances_count)
        return probas

    @staticmethod
    def _prune_least_probable(probas, threshold=0.05):
        remainder_probability = np.sum(probas[probas < threshold])
        probas[probas < 0.05] = 0
        still_positive_count = np.count_nonzero(probas)
        probas[probas > 0.0] += remainder_probability / still_positive_count
        return probas

    @staticmethod
    def _validate_args(history_embeddings, n):
        if history_embeddings is None or not len(history_embeddings):
            raise ValueError(
                "Cold-start Exception: history must contain at least one element"
            )
        if type(n) != int or n <= 0:
            raise ValueError(
                f"Request for {n} recommendations is invalid. N must be a positive int."
            )
        return

    def _estimate_cluster_probs(
        self, history_embeddings, method="square", pruning=True, prun_thr=0.05
    ):
        # Predict clusters to which each track is most likely belonging to
        cluster_num_preds = self.knn_clf.predict(history_embeddings.cpu())
        # Initialize cluster counts with zeros
        occurances_count = [0 for _ in range(len(self.knn_clf.classes_))]
        # Count how many each cluster num occured in preds
        occuring_cluster_nums, occuring_cluster_counts = np.unique(
            cluster_num_preds, return_counts=True
        )
        # Update occurances_count with counted occurances
        for num, count in zip(occuring_cluster_nums, occuring_cluster_counts):
            occurances_count[num] = count
        probas = Recommender._occurances2probs_strategy(occurances_count, method)
        if pruning and prun_thr is not None:
            probas = Recommender._prune_least_probable(probas, prun_thr)
        return probas

    def _get_new_recommendations(self, n, cluster_probs):
        cluster_probs_sampling = np.random.choice(
            len(self.knn_clf.classes_), size=n, p=cluster_probs
        )
        recommendations = [
            np.random.choice(self.clusters_tracks[cluster_num])
            for cluster_num in cluster_probs_sampling
        ]
        return recommendations

    def get_recommendations(self, history_embeddings, n):
        Recommender._validate_args(history_embeddings, n)
        cluster_probs = self._estimate_cluster_probs(history_embeddings)
        recommendations = self._get_new_recommendations(n, cluster_probs)
        return recommendations
