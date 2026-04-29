import logging
import numpy as np

try:
    from recommender import recommend_songs
except ModuleNotFoundError:
    from src.recommender import recommend_songs

logger = logging.getLogger(__name__)

MUSICGEN_MODEL = "facebook/musicgen-small"
_MAX_SCORE = 6.5  # genre(1.0) + mood(1.0) + energy(4.0) + acoustic(0.5)

_musicgen_pipeline = None  # lazy-loaded on first use


def confidence_pct(score: float) -> int:
    """Express a recommendation score as a percentage of the maximum possible score."""
    return round(max(0.0, min(score, _MAX_SCORE)) / _MAX_SCORE * 100)


def _select_musicgen_device():
    """CUDA if available, otherwise CPU. MPS skipped — EnCodec decoder hits a channel limit."""
    try:
        import torch
    except ImportError:
        return None

    if torch.cuda.is_available():
        return 0
    return -1


def _get_musicgen_pipeline():
    global _musicgen_pipeline
    if _musicgen_pipeline is None:
        from transformers import pipeline
        device = _select_musicgen_device()
        logger.info("Loading %s (device=%s)", MUSICGEN_MODEL, device)
        print(f"\n[MusicGen] Loading {MUSICGEN_MODEL} — first run downloads ~300MB...")
        if device is None:
            _musicgen_pipeline = pipeline("text-to-audio", model=MUSICGEN_MODEL)
        else:
            _musicgen_pipeline = pipeline("text-to-audio", model=MUSICGEN_MODEL, device=device)
        logger.info("Model ready")
        print("[MusicGen] Model ready.")
    return _musicgen_pipeline


def _build_music_prompt(user_prefs: dict) -> str:
    """Build a descriptive MusicGen prompt from structured user preferences."""
    genre = user_prefs.get("favorite_genre", "ambient")
    mood = user_prefs.get("favorite_mood", "calm")
    energy = float(user_prefs.get("target_energy", 0.5))
    acoustic = bool(user_prefs.get("likes_acoustic", False))

    if energy < 0.3:
        energy_desc, bpm = "very slow and gentle", "60 bpm"
    elif energy < 0.6:
        energy_desc, bpm = "moderate pace", "85 bpm"
    elif energy < 0.8:
        energy_desc, bpm = "upbeat and energetic", "120 bpm"
    else:
        energy_desc, bpm = "fast and intense", "160 bpm"

    texture = "acoustic and organic" if acoustic else "electronic and produced"
    prompt = f"{mood} {genre} music, {texture}, {energy_desc}, {bpm}"
    logger.info("Built prompt: %s", prompt)
    return prompt


def summarize_recommendation_vibe(recommendations: list, user_prefs: dict) -> dict:
    """Summarize top recommendations into a vibe profile for MusicGen."""
    if not recommendations:
        return {
            "genre": user_prefs.get("favorite_genre", "ambient"),
            "mood": user_prefs.get("favorite_mood", "calm"),
            "energy": float(user_prefs.get("target_energy", 0.5)),
            "tempo_bpm": 85,
            "acousticness": 1.0 if user_prefs.get("likes_acoustic", False) else 0.0,
            "source_titles": [],
        }

    top_recommendations = recommendations[:3]
    top_songs = [song for song, _score, _explanation in top_recommendations]
    lead_song = top_songs[0]
    genres = [song["genre"] for song in top_songs]
    moods = [song["mood"] for song in top_songs]

    def _most_common(values: list, fallback: str) -> str:
        counts = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        return max(counts, key=counts.get) if counts else fallback

    return {
        "genre": _most_common(genres, user_prefs.get("favorite_genre", "ambient")),
        "mood": _most_common(moods, user_prefs.get("favorite_mood", "calm")),
        "energy": float(lead_song["energy"]),
        "tempo_bpm": round(float(lead_song["tempo_bpm"])),
        "acousticness": float(lead_song["acousticness"]),
        "source_titles": [song["title"] for song in top_songs],
    }


def _build_music_prompt_from_recommendations(recommendations: list, user_prefs: dict) -> str:
    """Build a MusicGen prompt from the characteristics of top recommendations."""
    vibe = summarize_recommendation_vibe(recommendations, user_prefs)

    if vibe["energy"] < 0.3:
        energy_desc = "very slow and gentle"
    elif vibe["energy"] < 0.6:
        energy_desc = "moderate pace"
    elif vibe["energy"] < 0.8:
        energy_desc = "upbeat and energetic"
    else:
        energy_desc = "fast and intense"

    texture = "acoustic and organic" if vibe["acousticness"] >= 0.5 else "electronic and produced"
    prompt = (
        f"{vibe['mood']} {vibe['genre']} music, {texture}, "
        f"{energy_desc}, {vibe['tempo_bpm']} bpm"
    )
    logger.info("Built recommendation-grounded prompt: %s", prompt)
    return prompt


def build_song_prompt(song: dict) -> str:
    """Build a MusicGen prompt from a single song's attributes."""
    energy = float(song.get("energy", 0.5))
    if energy < 0.3:
        energy_desc = "very slow and gentle"
    elif energy < 0.6:
        energy_desc = "moderate pace"
    elif energy < 0.8:
        energy_desc = "upbeat and energetic"
    else:
        energy_desc = "fast and intense"
    texture = "acoustic and organic" if float(song.get("acousticness", 0)) >= 0.5 else "electronic and produced"
    return f"{song.get('mood', 'calm')} {song.get('genre', 'ambient')} music, {texture}, {energy_desc}, {song.get('tempo_bpm', 85):.0f} bpm"


def generate_audio_array(prompt: str, duration_seconds: int = 6):
    """Generate audio and return (array, sample_rate) without playing."""
    duration_seconds = max(5, min(15, duration_seconds))
    max_new_tokens = duration_seconds * 50

    logger.info("Generating %ds | prompt: %s", duration_seconds, prompt)
    pipe = _get_musicgen_pipeline()
    try:
        result = pipe(prompt, forward_params={"do_sample": True, "max_new_tokens": max_new_tokens})
    except NotImplementedError as exc:
        if "MPS device" not in str(exc):
            raise
        global _musicgen_pipeline
        _musicgen_pipeline = None
        logger.error("MPS kernel error — retrying on CPU")
        pipe = _get_musicgen_pipeline()
        result = pipe(prompt, forward_params={"do_sample": True, "max_new_tokens": max_new_tokens})

    if isinstance(result, list):
        result = result[0]
    audio = np.array(result["audio"])
    while audio.ndim > 1:
        audio = audio[0]
    return audio, int(result["sampling_rate"])


def _generate_and_play_music(prompt: str, duration_seconds: int = 8) -> str:
    duration_seconds = max(5, min(15, duration_seconds))
    max_new_tokens = duration_seconds * 50  # MusicGen small: ~50 tokens/sec

    logger.info("Generating %ds of audio | prompt: %s", duration_seconds, prompt)
    print(f"\n[MusicGen] Generating {duration_seconds}s: \"{prompt}\"")

    pipe = _get_musicgen_pipeline()
    try:
        result = pipe(prompt, forward_params={"do_sample": True, "max_new_tokens": max_new_tokens})
    except NotImplementedError as exc:
        if "MPS device" not in str(exc):
            raise

        global _musicgen_pipeline
        _musicgen_pipeline = None
        logger.error("MPS kernel error — retrying on CPU")
        print("[MusicGen] MPS hit a PyTorch kernel limit. Retrying on CPU...")
        pipe = _get_musicgen_pipeline()
        result = pipe(prompt, forward_params={"do_sample": True, "max_new_tokens": max_new_tokens})

    if isinstance(result, list):
        result = result[0]

    audio = np.array(result["audio"])
    while audio.ndim > 1:
        audio = audio[0]

    sr = int(result["sampling_rate"])
    actual_duration = len(audio) / sr
    logger.info("Audio ready: %.1fs at %dHz", actual_duration, sr)

    try:
        import sounddevice as sd
        print(f"[MusicGen] Playing {actual_duration:.1f}s of audio...")
        sd.play(audio.astype(np.float32), samplerate=sr)
        sd.wait()
        return f"Generated and played {actual_duration:.1f}s: \"{prompt}\""
    except ImportError:
        import scipy.io.wavfile as wavfile
        outfile = "vibe_preview.wav"
        logger.warning("sounddevice unavailable — saving to %s", outfile)
        wavfile.write(outfile, sr, audio.astype(np.float32))
        print(f"[MusicGen] Saved to {outfile} — open it to listen!")
        return f"Generated {actual_duration:.1f}s saved to {outfile}: \"{prompt}\""


def _format_recommendations(recommendations: list) -> str:
    if not recommendations:
        return "No matching songs found."
    lines = []
    for i, (song, score, _explanation) in enumerate(recommendations, 1):
        lines.append(
            f"{i}. \"{song['title']}\" by {song['artist']} "
            f"(genre: {song['genre']}, mood: {song['mood']}, "
            f"energy: {song['energy']:.2f}, score: {score:.2f})"
        )
    return "\n".join(lines)


def run_recommendation_with_audio(user_prefs: dict, songs: list, duration_seconds: int = 8) -> None:
    """Get top-5 song recommendations and generate a matching audio preview."""
    logger.info(
        "Session | genre=%s mood=%s energy=%.1f acoustic=%s",
        user_prefs.get("favorite_genre"),
        user_prefs.get("favorite_mood"),
        user_prefs.get("target_energy", 0.5),
        user_prefs.get("likes_acoustic", False),
    )

    recs = recommend_songs(user_prefs, songs, k=5)

    print("\nTop picks for you:")
    print("=" * 60)
    for i, (song, score, explanation) in enumerate(recs, 1):
        reasons = [r.strip() for r in explanation.split(";") if r.strip()]
        pct = confidence_pct(score)
        print(f"{i}. {song['title']} — {song['artist']}")
        print(f"   Score: {score:.2f} / {_MAX_SCORE}  (confidence: {pct}%)")
        print(f"   {', '.join(reasons)}")
    print("=" * 60)

    if recs:
        top_score = recs[0][1]
        logger.info("Top recommendation: %s (confidence: %d%%)", recs[0][0]["title"], confidence_pct(top_score))
        vibe = summarize_recommendation_vibe(recs, user_prefs)
        print(
            "Generated preview is based on your top recommendations: "
            + ", ".join(vibe["source_titles"])
        )
        print(
            "Shared vibe: "
            f"{vibe['mood']}, {vibe['genre']}, about {vibe['tempo_bpm']} bpm, "
            f"{'acoustic' if vibe['acousticness'] >= 0.5 else 'electronic'} texture."
        )

    prompt = _build_music_prompt_from_recommendations(recs, user_prefs)
    _generate_and_play_music(prompt, duration_seconds)
