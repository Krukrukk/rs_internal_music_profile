import os
from typing import Union

from fastapi.responses import JSONResponse  # type: ignore
from fastapi.encoders import jsonable_encoder  # type: ignore
import uvicorn  # type: ignore
from transformers import BertTokenizer, BertModel

from config.app import app
from logger import get_logger
from config import models
from config.globals import (
    SPOTIPY_CLIENT_ID,
    SPOTIPY_CLIENT_SECRET,
    SPOTIPY_REDIRECT_URI,
)
from manager import SpotifyManager
from embeddings import Embedder

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

logger = get_logger(__name__)
connector = SpotifyManager(
    SPOTIPY_CLIENT_ID,
    SPOTIPY_CLIENT_SECRET,
    SPOTIPY_REDIRECT_URI,
    "user-read-currently-playing user-read-recently-played",
    bert_model,
    bert_tokenizer,
)


@app.get("/", response_model=models.Status)
def check_server() -> JSONResponse:
    """Checks if server is functioning properly"""
    logger.info("Works")
    return JSONResponse({"success": True}, 200)


@app.get("/song/data/raw", response_model=models.Songs)
def get_data_song_raw(track_uri: str) -> JSONResponse:
    response = connector.get_data_song_raw(track_uri)
    return JSONResponse(jsonable_encoder(response), 200)


@app.post("/songs/data/raw", response_model=models.Songs)
def get_data_song_raw(track_data: models.TrackData) -> JSONResponse:
    response = [
        connector.get_data_song_raw(track_uri) for track_uri in track_data.track_uris
    ]
    return JSONResponse(jsonable_encoder(response), 200)


@app.get("/song/data/preprocessed", response_model=models.Songs)
def get_data_song_preprocessed(track_uri: str) -> JSONResponse:
    response = connector.get_data_songs_preprocessed([track_uri])
    return JSONResponse(jsonable_encoder(response), 200)


@app.post("/songs/data/preprocessed", response_model=models.Songs)
def get_data_song_preprocessed(track_data: models.TrackData) -> JSONResponse:
    response = connector.get_data_songs_preprocessed(track_data.track_uris)
    return JSONResponse(jsonable_encoder(response), 200)


@app.get("/song/data/embeddings", response_model=models.Songs)
def get_data_song_preprocessed(track_uri: str) -> JSONResponse:
    response = connector.generate_songs_embeddings([track_uri])
    logger.info(response)
    return JSONResponse({"embeddings": response.tolist()}, 200)


@app.post("/songs/data/embeddings", response_model=models.Songs)
def get_data_song_preprocessed(track_data: models.TrackData) -> JSONResponse:
    response = connector.generate_songs_embeddings(track_data.track_uris)
    logger.info(response)
    return JSONResponse({"embeddings": response.tolist()}, 200)


@app.get("/user/current", response_model=models.Songs)
def get_current_song() -> JSONResponse:
    response = connector.get_current_song()
    return JSONResponse(jsonable_encoder(response), 200)


@app.get("/user/history", response_model=models.Songs)
def get_current_songs(n: int = 50) -> JSONResponse:
    response = connector.get_history_songs(n)
    return JSONResponse(jsonable_encoder(response), 200)


@app.get("/user/session", response_model=models.Songs)
def get_session(n: int = 20, margin: int = 300) -> JSONResponse:
    """
    This endpint returns user listening session based on listening time

    n : int = 50
        Max number of songs in session
    margin : int = 30
        Maximum pause in songs treat them as a session in seconds
    """
    response = connector.get_session(n, margin=margin)
    return JSONResponse(jsonable_encoder(response), 200)


if __name__ == "__main__":

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
