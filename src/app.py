"""
EmotionLift: Emotion-Based Music Recommender — Streamlit GUI

Run:
    streamlit run src/app.py
"""

import sys
from pathlib import Path

# Allow imports from src/ when launched from project root
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from transformers import pipeline as hf_pipeline
from recommender import load_songs, recommend_songs
from therapy import llm_classify_emotion
from agent import confidence_pct
from voice_input import transcribe_audio

MEDIA_DIR = Path(__file__).parent.parent / "data" / "media"


@st.cache_resource(show_spinner="Loading speech recognition model (one-time)…")
def get_whisper():
    return hf_pipeline("automatic-speech-recognition", model="openai/whisper-base", device=-1)


@st.cache_resource(show_spinner="Loading emotion classifier (one-time)…")
def get_emotion_classifier():
    return hf_pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        device=-1,
    )


def render_recommendations(recs):
    """Render the scored recommendation list with audio players."""
    for i, (song, score, explanation) in enumerate(recs, 1):
        pct = confidence_pct(score)
        mp3_path = MEDIA_DIR / song["file"] if song.get("file") else None
        with st.container():
            cols = st.columns([0.05, 0.50, 0.20, 0.25])
            cols[0].write(f"**{i}.**")
            cols[1].write(f"**{song['title']}** — *{song['artist']}*")
            cols[2].write(f"`{song['genre']}` / `{song['mood']}`")
            cols[3].progress(pct / 100, text=f"{pct}% match")
            if mp3_path and mp3_path.exists():
                st.audio(str(mp3_path), format="audio/mp3", autoplay=(i == 1))


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Music Therapy", page_icon="🎵", layout="centered")

# ── Load songs once ───────────────────────────────────────────────────────────
@st.cache_data
def get_songs():
    csv = Path(__file__).parent.parent / "data" / "songs.csv"
    return load_songs(str(csv))

songs = get_songs()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🎵 EmotionLift: Emotion-Based Music Recommender")
st.caption("Speak how you're feeling and we'll find music to help.")

st.markdown("Press **Record** and speak how you're feeling — we'll find the right music.")
audio_input = st.audio_input("Speak now", label_visibility="collapsed")

if audio_input is not None:
    with st.spinner("Transcribing your voice…"):
        transcribed = transcribe_audio(audio_input.read(), get_whisper())

    st.info(f'You said: "{transcribed}"')

    with st.spinner("Detecting emotion…"):
        prefs = llm_classify_emotion(transcribed, get_emotion_classifier())

    emotion_label = prefs.pop("_emotion_label", "neutral")
    emotion_conf = prefs.pop("_emotion_confidence", 0.0)

    st.caption(f"Detected emotion: **{emotion_label}** ({emotion_conf:.0%} confidence)")

    st.divider()
    st.subheader("Your therapeutic vibe")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Genre", prefs["favorite_genre"].title())
    col2.metric("Mood", prefs["favorite_mood"].title())
    col3.metric("Energy", f"{prefs['target_energy']:.0%}")
    col4.metric("Texture", "Acoustic" if prefs["likes_acoustic"] else "Electronic")

    st.divider()
    st.subheader("Recommended songs")
    recs = recommend_songs(prefs, songs, k=5)
    render_recommendations(recs)
