import os
import pickle
from PIL import Image
import streamlit
import requests
import random
import streamlit.components.v1 as components
from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth
from config.globals import (
    SPOTIPY_CLIENT_ID,
    SPOTIPY_CLIENT_SECRET,
    SPOTIPY_REDIRECT_URI,
)
import asyncio
import logging

# from logger import get_logger

logger = logging.getLogger("__name__")
logger.setLevel(logging.WARNING)
WITH_IMG = 640


def write_authorization_url(manager):
    authorization_url = manager.get_authorize_url()
    return authorization_url


def write_access_token(manager, code):
    token = manager.get_access_token(code, as_dict=False, check_cache=False)
    return token


class SentimentApp:
    def __init__(self, auth_manager, base_url="http://localhost:8080/"):
        self.auth_manager = auth_manager
        self.base_url = base_url

    def __call__(self) -> None:
        streamlit.image("img/dyza.png")
        streamlit.title("Spot your taste.")
        streamlit.markdown(
            "Hi! If you want to get access to this app contact us on discord **@piniu9898#0636**, **@Mateusz Czyż#1431**, **@Maciej Małecki#6709**"
        )
        streamlit.markdown(
            "We used triplet network to generate song embeddigs and k-means with extra steps to provide you best recommendations! <br>"
            "Currently we have **180k** songs in our dataset. Of course we didn't index all spotify songs, but it's probably enough to find something for you!",
            unsafe_allow_html=True,
        )

        if (
            "auth_token" not in streamlit.session_state
            or not streamlit.session_state["auth_token"]
        ):
            try:
                auth_url = write_authorization_url(self.auth_manager)
                code = streamlit.experimental_get_query_params()["code"]
                try:
                    auth_token = write_access_token(self.auth_manager, code)
                    streamlit.session_state["auth_token"] = auth_token
                except Exception as e:
                    streamlit.write(
                        f"""<h1>
                        This account is not allowed or page was refreshed.
                        Please try again: <a target="_self"
                        href="{auth_url}">url</a></h1>""",
                        unsafe_allow_html=True,
                    )
            except:
                streamlit.write(
                    f"""<h2>
                Please log in using this <a target="_self"
                href="{auth_url}">url</a></h2>""",
                    unsafe_allow_html=True,
                )
        if (
            "auth_token" in streamlit.session_state
            and streamlit.session_state["auth_token"]
        ):
            with streamlit.form(key="my_form"):
                submit_button = streamlit.form_submit_button(label="Refresh")

                spotify = Spotify(auth=streamlit.session_state["auth_token"])

                ################# WELCOME ###################
                streamlit.markdown(f"Hi {spotify.current_user()['display_name']}!")
                ####################################
                ################# SHOW CURRENT ###################
                streamlit.markdown("Currently playing")
                response = requests.get(
                    f"{self.base_url}user/current",
                    params={"token": streamlit.session_state["auth_token"]},
                ).json()
                if not response:
                    streamlit.markdown("Currently not playing anything")
                else:
                    title = response["name"]
                    artists = response["artists"]
                    streamlit.markdown(
                        f"Wow, a really nice playlist! You current session is:"
                    )
                    response = requests.get(
                        f"{self.base_url}user/session",
                        params={"token": streamlit.session_state["auth_token"]},
                    ).json()

                    for song in response:
                        i_frame = components.iframe(
                            src=f"https://open.spotify.com/embed/track/{song['uri'].split(':')[-1]}?utm_source=generator"
                        )

                #################################################
            N = 20
            with streamlit.form(key="my_form_recs"):
                streamlit.markdown(
                    f"Do you need some recommendations based on your {N} last songs?"
                )
                submit_button_recs = streamlit.form_submit_button(label="Generate")
                if submit_button_recs:
                    ################# SHOW RECOMMENDATIONS #######################
                    streamlit.markdown(
                        f"Basing on your {N} last songs we recommend you:"
                    )
                    response = requests.get(
                        f"{self.base_url}songs/data/recommendations",
                        params={
                            "token": streamlit.session_state["auth_token"],
                            "n": 10,
                            "hist": N,
                            "_type": "hist",
                        },
                    ).json()
                    if not response:
                        streamlit.markdown("Oops, somethig went wrong!")
                    else:
                        # streamlit.write(response)
                        for uri in response:

                            i_frame = components.iframe(
                                src=f"https://open.spotify.com/embed/track/{uri.split(':')[-1]}?utm_source=generator"
                            )
                    ###############################################
            with streamlit.form(key="my_form_recs_sess"):
                streamlit.markdown(
                    "Do you need some recommendations based on your session?"
                )
                submit_button_recs_sess = streamlit.form_submit_button(label="Generate")
                if submit_button_recs_sess:
                    ################# SHOW RECOMMENDATIONS #######################
                    N = 20
                    streamlit.markdown(f"Basing on your current session")
                    response = requests.get(
                        f"{self.base_url}songs/data/recommendations",
                        params={
                            "token": streamlit.session_state["auth_token"],
                            "n": 10,
                            "hist": N,
                            "_type": "session",
                        },
                    ).json()
                    if not response:
                        streamlit.markdown("Oops, somethig went wrong!")
                    else:
                        # streamlit.write(response)
                        for uri in response:

                            i_frame = components.iframe(
                                src=f"https://open.spotify.com/embed/track/{uri.split(':')[-1]}?utm_source=generator"
                            )
                    ###############################################

        streamlit.markdown(
            "**Authors**<br>"
            "[Mateusz Czyż](https://github.com/CzyzuM/)<br>"
            "[Maciej Małecki](https://github.com/Krukrukk)<br>"
            "[Patryk Szelewski](https://github.com/pszelew/)<br>"
            "[Dyza Boys - Nekeca](https://www.youtube.com/watch?v=GHDhA72nEEo)",
            unsafe_allow_html=True,
        )

        streamlit.markdown(
            "Source code<br>"
            "[Project on GitHub](https://github.com/Krukrukk/rs_internal_music_profile)",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    auth_manager = SpotifyOAuth(
        client_id=SPOTIPY_CLIENT_ID,
        client_secret=SPOTIPY_CLIENT_SECRET,
        redirect_uri=SPOTIPY_REDIRECT_URI,
        scope="user-read-currently-playing user-read-recently-played",
        show_dialog=True,
    )
    app = SentimentApp(auth_manager)
    app()
