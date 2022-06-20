### IMPORTS

# General
import pandas as pd
import re

# Lyric processing
import nltk
from nltk.corpus import stopwords
from collections import Counter
# nltk.download('stopwords')

### LOAD DATA

# Read in data
data = pd.read_csv("spotify_songs.csv")
# Select the relevant columns
keep_cols = [x for x in data.columns if not x.startswith("track") and not x.startswith("playlist")]
keep_cols.append("playlist_genre")
df = data[keep_cols].copy()
# Filter to only English and remove Latin genre (leakage)
subdf = df[(df.language == "en") & (df.playlist_genre != "latin")].copy().drop(columns = "language")



### PRE-PROCESS LYRICS

# Tidy up lyric text, only include lowercase letters
pattern = r"[^a-zA-Z ]"
subdf.lyrics = subdf.lyrics.apply(lambda x: re.sub(pattern, "", x.lower()))
# Remove stopwords
subdf.lyrics = subdf.lyrics.apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords.words("english"))]))

# Find how often words appear across all lyrics
# Join all lyrics
all_text = " ".join(subdf.lyrics)
# Find counts of each word
word_count = Counter(all_text.split())
# Filter out words that don't appear in at least 200 songs
keep_words = [k for k, v in word_count.items() if v > 200]
# Create copy to work with
lyricdf = subdf.copy().reset_index(drop=True)
# Neaten up columns
lyricdf.columns = ["audio_"+ x if not x in ["lyrics", "playlist_genre"] else x for x in lyricdf.columns]
# Get lyric counts
lyricdf.lyrics = lyricdf.lyrics.apply(lambda x: Counter([word for word in x.split() if word in keep_words]))
# Unpack lyric counts / Counter dict into columns in df
unpacked_lyrics = pd.DataFrame.from_records(lyricdf.lyrics).add_prefix("lyrics_")
# change NAs to 0
unpacked_lyrics = unpacked_lyrics.fillna(0) 
# Concatenate and drop original lyric col
lyricdf = pd.concat([lyricdf, unpacked_lyrics], axis = 1).drop(columns = "lyrics")
# Rearrange lyric cols to be alphabetical
reordered_cols = [col for col in lyricdf.columns if not col.startswith("lyrics_")] + sorted([col for col in lyricdf.columns if col.startswith("lyrics_")])
lyricdf = lyricdf[reordered_cols]

# Save to csv
lyricdf.to_csv("music_data.csv", index = False)