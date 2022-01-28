from http.client import ImproperConnectionState
import imp
import json
from datetime import datetime, timedelta
import re
from typing import Union

from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel

from features import FeatureProcessor
from recommender import Recommender
from embeddings import Embedder
from logger import get_logger

logger = get_logger(__name__)


class SpotifyManager(FeatureProcessor, Embedder):
    def __init__(
        self,
        scope: str,
        model: BertModel,
        tokenizer: BertTokenizer,
        recommender: Recommender,
        embedder_kwargs: dict = {},
    ) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        FeatureProcessor.__init__(self, model, tokenizer, device)
        Embedder.__init__(self, device, embedder_kwargs)
        self.recommender = recommender

    @staticmethod
    def _preprocess_history_songs(songs: dict) -> list:
        """
        Preprocesses current songs from given songs json string.

        songs : dict
            Dict with songs from spotify API
        to_json : bool
            If true, will jsonify the response

        """
        tracks = [(song["track"], song["played_at"]) for song in songs["items"]]
        return [
            {
                "uri": track["uri"],
                "name": track["name"],
                "artists": [artist["name"] for artist in track["artists"]],
                "duration_ms": track["duration_ms"],
                "played_at": datetime.strptime(played_at, "%Y-%m-%dT%H:%M:%S.%fZ"),
                "end_est": datetime.strptime(played_at, "%Y-%m-%dT%H:%M:%S.%fZ")
                + timedelta(milliseconds=track["duration_ms"]),
            }
            for track, played_at in tracks
        ]

    @staticmethod
    def _preprocess_current_song(song: dict) -> list:
        """
        Preprocesses current song from given songs json string.

        songs : dict
            Dict with songs from spotify API
        to_json : bool
            If true, will jsonify the response

        """
        return {
            "uri": song["item"]["uri"],
            "name": song["item"]["name"],
            "artists": [artist["name"] for artist in song["item"]["artists"]],
            "duration_ms": song["item"]["duration_ms"],
            "played_at": datetime.utcfromtimestamp(song["timestamp"] / 1000),
            "end_est": datetime.utcfromtimestamp(song["timestamp"] / 1000)
            + timedelta(milliseconds=song["item"]["duration_ms"]),
            "progress_ms": song["progress_ms"],
        }

    # ----------------------------------------------------------------
    # User playling logs
    # ----------------------------------------------------------------

    def get_current_song(self, token: str) -> dict:
        client = Spotify(auth=token)
        response = client.currently_playing()

        if response:
            return self._preprocess_current_song(response)
        else:
            return None

    def get_history_songs(self, token, n: int) -> list[dict]:
        client = Spotify(auth=token)
        response = client.current_user_recently_played(limit=n)
        return self._preprocess_history_songs(response)

    def get_session(self, token, n: int = 50, margin: int = 300) -> list[dict]:
        """
        This method returns user listening session based on listening time

        n : int = 50
            Max number of songs in session
        margin : int = 300
            Maximum pause in songs treat them as a session in seconds
        """
        current_song = self.get_current_song(token)
        history_songs = self.get_history_songs(token, n)
        session_songs = [current_song]

        for song in history_songs:
            delta = session_songs[-1]["played_at"] - song["end_est"]
            logger.info(delta)
            if delta > timedelta(seconds=margin):
                return session_songs
            session_songs.append(song)
        return session_songs

    # ----------------------------------------------------------------
    # Tracks data
    # ----------------------------------------------------------------

    def get_data_song_raw(self, token: str, track_uri: str) -> dict:
        """
        Fetches song sumeric, text and sequence features by giving it's track_uri

        track_uri : str
        """
        client = Spotify(auth=token)
        features = client.audio_features(track_uri)[0]
        general_data = client.track(track_uri)
        sections = client.audio_analysis(track_uri)["sections"]

        album_name = general_data["album"]["name"]
        track_name = general_data["name"]
        artist_name = " ".join([artist["name"] for artist in general_data["artists"]])

        features["album_name"] = album_name
        features["track_name"] = track_name
        features["artist_name"] = artist_name

        features["analysis_sections"] = sections
        return features

    def get_data_songs_preprocessed(
        self, token: str, tracks_uris: list[str], to_dict=True
    ) -> dict:
        data_raw = [
            self.get_data_song_raw(token, track_uri) for track_uri in tracks_uris
        ]
        df = pd.DataFrame(data_raw)
        self.preprocess_features(df)
        if to_dict:
            return df.to_dict("records")
        return df

    def generate_songs_embeddings(self, token, tracks_uris: list[str]):
        df = self.get_data_songs_preprocessed(token, tracks_uris, False)
        numeric_features = torch.tensor(df[list(self.numeric_features.keys())].values)
        text_features = torch.stack(list(df["text_embeddings"]))
        seq_features = torch.stack(list(df[f"{self.section_feature['name']}_seq"]))

        embedding = self.get_embedding(numeric_features, text_features, seq_features)
        return embedding

    # ----------------------------------------------------------------
    # Recommender
    # ----------------------------------------------------------------

    def get_recommendations(
        self, token: str, n: int = 10, hist: int = 20, _type: str = "hist"
    ):
        """
        _type : str
            Possible values: _hist / session
        """
        if _type == "hist":
            history = self.get_history_songs(token, hist)

        else:
            history = self.get_session(token, n)

        history_embeddings = self.generate_songs_embeddings(
            token, map(lambda song: song["uri"], history)
        )

        recommendations = self.recommender.get_recommendations(history_embeddings, n)
        return recommendations
