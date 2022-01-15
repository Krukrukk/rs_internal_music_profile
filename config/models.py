from enum import Enum
from os import stat
from pydantic import BaseModel


class Status(BaseModel):
    status: str


class Song(BaseModel):
    todo: str


class Songs(BaseModel):
    songs: list[Song]


class TrackData(BaseModel):
    track_uris: list[str]
