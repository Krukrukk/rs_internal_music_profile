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
from embeddings import Embedder
from logger import get_logger

logger = get_logger(__name__)


class SpotifyManager(FeatureProcessor, Embedder):
    def __init__(
        self,
        spotify_client_id: str,
        spotify_client_secret: str,
        spotify_redirect_uri: str,
        scope: str,
        model: BertModel,
        tokenizer: BertTokenizer,
        embedder_kwargs: dict = {},
    ) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        FeatureProcessor.__init__(self, model, tokenizer, device)
        Embedder.__init__(self, device, embedder_kwargs)
        self.scope = scope
        self.client = Spotify(
            auth_manager=SpotifyOAuth(
                client_id=spotify_client_id,
                client_secret=spotify_client_secret,
                redirect_uri=spotify_redirect_uri,
                scope=scope,
            )
        )

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

    def get_current_song(self) -> dict:
        response = self.client.currently_playing()
        return self._preprocess_current_song(response)

    def get_history_songs(self, n: int) -> list[dict]:
        response = self.client.current_user_recently_played(limit=n)
        return self._preprocess_history_songs(response)

    def get_session(self, n: int = 50, margin: int = 300) -> list[dict]:
        """
        This method returns user listening session based on listening time

        n : int = 50
            Max number of songs in session
        margin : int = 300
            Maximum pause in songs treat them as a session in seconds
        """
        current_song = self.get_current_song()
        history_songs = self.get_history_songs(n)
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

    def get_data_song_raw(self, track_uri: str) -> dict:
        """
        Fetches song sumeric, text and sequence features by giving it's track_uri

        track_uri : str
        """
        features = self.client.audio_features(track_uri)[0]
        general_data = self.client.track(track_uri)
        sections = self.client.audio_analysis(track_uri)["sections"]

        album_name = general_data["album"]["name"]
        track_name = general_data["name"]
        artist_name = " ".join([artist["name"] for artist in general_data["artists"]])

        features["album_name"] = album_name
        features["track_name"] = track_name
        features["artist_name"] = artist_name

        features["analysis_sections"] = sections
        return features

    def get_data_songs_preprocessed(self, tracks_uris: list[str], to_dict=True) -> dict:
        data_raw = [self.get_data_song_raw(track_uri) for track_uri in tracks_uris]
        df = pd.DataFrame(data_raw)
        self.preprocess_features(df)
        if to_dict:
            return df.to_dict("records")
        return df

    def generate_songs_embeddings(self, tracks_uris: list[str]):
        df = self.get_data_songs_preprocessed(tracks_uris, False)
        numeric_features = torch.tensor(df[list(self.numeric_features.keys())].values)
        text_features = torch.stack(list(df["text_embeddings"]))
        seq_features = torch.stack(list(df[f"{self.section_feature['name']}_seq"]))

        embedding = self.get_embedding(numeric_features, text_features, seq_features)
        return embedding
