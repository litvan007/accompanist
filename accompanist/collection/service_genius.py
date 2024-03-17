from contextlib import redirect_stdout
from os import devnull

import lyricsgenius

from accompanist.config import settings


def get_lyrics_from_genius(artist_name: str, song_name: str):
    genius = lyricsgenius.Genius(settings.GENIUS_CLIENT_ACCESS_TOKEN)

    # Keep section headers (e.g. [Chorus]) in lyrics
    genius.remove_section_headers = False

    with open(devnull, "w") as fnull:
        # disable `print`s in this function
        with redirect_stdout(fnull):
            song = genius.search_song(
                title=song_name,
                artist=artist_name,
                get_full_info=False,
            )
    lyrics = song.lyrics

    # `lyricsgenius` parser gives some extra words at the beginning and at the
    # end => remove extra words (maybe one day it will be fixed)
    first_newline = lyrics.find("\n")
    lyrics = lyrics[first_newline + 1 :]
    if lyrics.endswith("Embed"):
        lyrics = lyrics[: -len("Embed")]
    return lyrics
