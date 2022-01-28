import logging

logging.getLogger("requests").setLevel(logging.WARNING)

import os
from typing import Union, Optional

from fastapi.responses import JSONResponse  # type: ignore
from fastapi.encoders import jsonable_encoder  # type: ignore
import uvicorn  # type: ignore
from transformers import BertTokenizer, BertModel
from fastapi_sessions.backends.implementations import InMemoryBackend
from fastapi.responses import RedirectResponse

from config.app import app
from logger import get_logger
from config import models
from manager import SpotifyManager
from recommender import Recommender
from embeddings import Embedder

logger = logging.getLogger("spotipy.client")
logger.setLevel("WARNING")


BERT_TOKENIER_PATH = "embeddings/pretrained/BertTokenizer"
BERT_MODEL_PATH = "embeddings/pretrained/BertModel"


if os.path.exists(BERT_TOKENIER_PATH):
    bert_tokenizer = BertTokenizer.from_pretrained(BERT_TOKENIER_PATH)
else:
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-uncased")
    bert_tokenizer.save_pretrained("embeddings/pretrained/BertTokenizer")
if os.path.exists(BERT_MODEL_PATH):
    bert_model = BertModel.from_pretrained("embeddings/pretrained/BertModel")
else:
    bert_model = BertModel.from_pretrained("bert-base-multilingual-uncased")
    bert_model.save_pretrained("embeddings/pretrained/BertModel")


recommender = Recommender()

connector = SpotifyManager(
    "user-read-currently-playing user-read-recently-played",
    bert_model,
    bert_tokenizer,
    recommender,
)


@app.get("/", response_model=models.Status)
def check_server() -> JSONResponse:
    """Checks if server is functioning properly"""
    logger.info("Works")
    return JSONResponse({"success": True}, 200)


@app.get("/song/data/raw", response_model=models.Songs)
def get_data_song_raw(token, track_uri: str) -> JSONResponse:
    response = connector.get_data_song_raw(token, track_uri)
    return JSONResponse(jsonable_encoder(response), 200)


@app.post("/songs/data/raw", response_model=models.Songs)
def get_data_song_raw(token, track_data: models.TrackData) -> JSONResponse:
    response = [
        connector.get_data_song_raw(token, track_uri)
        for track_uri in track_data.track_uris
    ]
    return JSONResponse(jsonable_encoder(response), 200)


@app.get("/songs/data/recommendations", response_model=models.Songs)
def recommend_songs(
    token, n: int = 10, hist: int = 20, _type: str = "hist"
) -> JSONResponse:
    """
    _type : str
        Possible values: _hist / session
    """
    response = connector.get_recommendations(token, n, hist, _type)
    return JSONResponse(jsonable_encoder(response), 200)


@app.get("/song/data/preprocessed", response_model=models.Songs)
def get_data_song_preprocessed(token, track_uri: str) -> JSONResponse:
    response = connector.get_data_songs_preprocessed(token, [track_uri])
    return JSONResponse(jsonable_encoder(response), 200)


@app.post("/songs/data/preprocessed", response_model=models.Songs)
def get_data_song_preprocessed(token, track_data: models.TrackData) -> JSONResponse:
    response = connector.get_data_songs_preprocessed(token, track_data.track_uris)
    return JSONResponse(jsonable_encoder(response), 200)


@app.get("/song/data/embeddings", response_model=models.Songs)
def get_data_song_preprocessed(token, track_uri: str) -> JSONResponse:
    response = connector.generate_songs_embeddings(token, [track_uri])
    return JSONResponse({"embeddings": response.tolist()}, 200)


@app.post("/songs/data/embeddings", response_model=models.Songs)
def get_data_song_preprocessed(token, track_data: models.TrackData) -> JSONResponse:
    response = connector.generate_songs_embeddings(token, track_data.track_uris)
    return JSONResponse({"embeddings": response.tolist()}, 200)


@app.get("/user/current", response_model=models.Songs)
def get_current_song(token: str) -> JSONResponse:
    response = connector.get_current_song(token)
    return JSONResponse(jsonable_encoder(response), 200)


@app.get("/user/history", response_model=models.Songs)
def get_current_songs(token, n: int = 50) -> JSONResponse:
    response = connector.get_history_songs(token, n)
    return JSONResponse(jsonable_encoder(response), 200)


@app.get("/user/session", response_model=models.Songs)
def get_session(token: str, n: int = 20, margin: int = 300) -> JSONResponse:
    """
    This endpint returns user listening session based on listening time

    n : int = 50
        Max number of songs in session
    margin : int = 300
        Maximum pause in songs treat them as a session in seconds
    """
    response = connector.get_session(token, n, margin=margin)
    return JSONResponse(jsonable_encoder(response), 200)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", type=int, default=8080, help="Port number")
    args = parser.parse_args()
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=args.p,
        reload=False,
    )
