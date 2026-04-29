import sys
import numpy as np
from unittest.mock import MagicMock, patch
from src.agent import (
    _format_recommendations,
    _build_music_prompt,
    _build_music_prompt_from_recommendations,
    _generate_and_play_music,
    run_recommendation_with_audio,
    confidence_pct,
    summarize_recommendation_vibe,
)
from src.recommender import recommend_songs


SAMPLE_SONGS = [
    {
        "id": 1,
        "title": "Midnight Coding",
        "artist": "LoRoom",
        "genre": "lofi",
        "mood": "chill",
        "energy": 0.42,
        "tempo_bpm": 75.0,
        "valence": 0.5,
        "danceability": 0.5,
        "acousticness": 0.71,
    },
    {
        "id": 2,
        "title": "Gym Hero",
        "artist": "Max Pulse",
        "genre": "pop",
        "mood": "intense",
        "energy": 0.93,
        "tempo_bpm": 140.0,
        "valence": 0.8,
        "danceability": 0.8,
        "acousticness": 0.05,
    },
]


def test_format_recommendations_returns_numbered_list():
    recs = recommend_songs(
        {"favorite_genre": "lofi", "favorite_mood": "chill", "target_energy": 0.4, "likes_acoustic": True},
        SAMPLE_SONGS,
        k=2,
    )
    result = _format_recommendations(recs)
    assert "1." in result
    assert "2." in result
    assert "Midnight Coding" in result or "Gym Hero" in result


def test_build_music_prompt_includes_genre_and_mood():
    prefs = {"favorite_genre": "lofi", "favorite_mood": "chill", "target_energy": 0.4, "likes_acoustic": True}
    prompt = _build_music_prompt(prefs)
    assert "lofi" in prompt
    assert "chill" in prompt
    assert "acoustic" in prompt


def test_build_music_prompt_high_energy_is_fast():
    prefs = {"favorite_genre": "metal", "favorite_mood": "intense", "target_energy": 0.9, "likes_acoustic": False}
    prompt = _build_music_prompt(prefs)
    assert "metal" in prompt
    assert "160 bpm" in prompt
    assert "electronic" in prompt or "produced" in prompt


def test_summarize_recommendation_vibe_uses_top_ranked_songs():
    prefs = {"favorite_genre": "lofi", "favorite_mood": "chill", "target_energy": 0.4, "likes_acoustic": True}
    recs = recommend_songs(prefs, SAMPLE_SONGS, k=2)

    vibe = summarize_recommendation_vibe(recs, prefs)

    assert vibe["genre"] == "lofi"
    assert vibe["mood"] == "chill"
    assert vibe["source_titles"][0] == "Midnight Coding"


def test_build_music_prompt_from_recommendations_reflects_top_songs():
    prefs = {"favorite_genre": "lofi", "favorite_mood": "chill", "target_energy": 0.4, "likes_acoustic": True}
    recs = recommend_songs(prefs, SAMPLE_SONGS, k=2)

    prompt = _build_music_prompt_from_recommendations(recs, prefs)

    assert "lofi" in prompt
    assert "chill" in prompt
    assert "75 bpm" in prompt
    assert "acoustic" in prompt


def test_generate_and_play_music_calls_pipeline_with_prompt():
    fake_audio = np.zeros((1, 1, 32000 * 5), dtype=np.float32)
    mock_pipe = MagicMock(return_value={"audio": fake_audio, "sampling_rate": 32000})
    mock_sd = MagicMock()

    with patch("src.agent._get_musicgen_pipeline", return_value=mock_pipe), \
         patch.dict(sys.modules, {"sounddevice": mock_sd}):
        result = _generate_and_play_music("calm lo-fi piano, 70 bpm", 5)

    mock_pipe.assert_called_once()
    assert "calm lo-fi piano" in mock_pipe.call_args[0][0]
    assert "Generated" in result


def test_run_recommendation_with_audio_calls_musicgen():
    fake_audio = np.zeros((1, 1, 32000 * 8), dtype=np.float32)
    mock_pipe = MagicMock(return_value={"audio": fake_audio, "sampling_rate": 32000})
    mock_sd = MagicMock()

    prefs = {"favorite_genre": "lofi", "favorite_mood": "chill", "target_energy": 0.4, "likes_acoustic": True}

    with patch("src.agent._get_musicgen_pipeline", return_value=mock_pipe), \
         patch.dict(sys.modules, {"sounddevice": mock_sd}):
        run_recommendation_with_audio(prefs, SAMPLE_SONGS)

    mock_pipe.assert_called_once()
    generated_prompt = mock_pipe.call_args[0][0]
    assert "lofi" in generated_prompt
    assert "chill" in generated_prompt
    assert "75 bpm" in generated_prompt


def test_confidence_pct_range():
    """Confidence is always 0–100 and scales with score."""
    assert confidence_pct(0.0) == 0
    assert confidence_pct(6.5) == 100
    assert 0 < confidence_pct(3.25) < 100
    # Perfect match on a known profile should be high confidence
    recs = recommend_songs(
        {"favorite_genre": "lofi", "favorite_mood": "chill", "target_energy": 0.4, "likes_acoustic": True},
        SAMPLE_SONGS,
        k=1,
    )
    top_score = recs[0][1]
    assert confidence_pct(top_score) >= 50


def test_unknown_genre_still_returns_results():
    """Unknown genre degrades gracefully — still returns songs, just lower confidence."""
    recs = recommend_songs(
        {"favorite_genre": "bossa nova", "favorite_mood": "chill", "target_energy": 0.4, "likes_acoustic": True},
        SAMPLE_SONGS,
        k=2,
    )
    assert len(recs) == 2
    # No genre points awarded — confidence should be lower than a matched profile
    unknown_conf = confidence_pct(recs[0][1])
    recs_known = recommend_songs(
        {"favorite_genre": "lofi", "favorite_mood": "chill", "target_energy": 0.4, "likes_acoustic": True},
        SAMPLE_SONGS,
        k=1,
    )
    known_conf = confidence_pct(recs_known[0][1])
    assert unknown_conf < known_conf
