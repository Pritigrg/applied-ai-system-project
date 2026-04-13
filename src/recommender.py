from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import csv
from pathlib import Path

@dataclass
class Song:
    """Represents a song and its attributes."""
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """Represents a user's taste preferences."""
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """OOP implementation of the recommendation logic."""
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Return the top-k songs for a user sorted by score."""
        scored: List[Tuple[float, Song]] = []
        for song in self.songs:
            song_dict = {
                "id": song.id,
                "title": song.title,
                "artist": song.artist,
                "genre": song.genre,
                "mood": song.mood,
                "energy": song.energy,
                "tempo_bpm": song.tempo_bpm,
                "valence": song.valence,
                "danceability": song.danceability,
                "acousticness": song.acousticness,
            }
            user_prefs = {
                "favorite_genre": user.favorite_genre,
                "favorite_mood": user.favorite_mood,
                "target_energy": user.target_energy,
                "likes_acoustic": user.likes_acoustic,
            }
            score, _ = score_song(user_prefs, song_dict)
            scored.append((score, song))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [song for _, song in scored[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Return a human-readable explanation for one song recommendation."""
        song_dict = {
            "id": song.id,
            "title": song.title,
            "artist": song.artist,
            "genre": song.genre,
            "mood": song.mood,
            "energy": song.energy,
            "tempo_bpm": song.tempo_bpm,
            "valence": song.valence,
            "danceability": song.danceability,
            "acousticness": song.acousticness,
        }
        user_prefs = {
            "favorite_genre": user.favorite_genre,
            "favorite_mood": user.favorite_mood,
            "target_energy": user.target_energy,
            "likes_acoustic": user.likes_acoustic,
        }
        score, reasons = score_song(user_prefs, song_dict)
        return f"Score {score:.2f}: " + "; ".join(reasons)

def load_songs(csv_path: str = "data/songs.csv") -> List[Dict]:
    """Load songs from a CSV file into a list of dictionaries."""
    songs: List[Dict] = []
    path = Path(csv_path)
    if not path.is_absolute() and not path.exists():
        # Resolve relative paths from the project root when run from src/.
        path = Path(__file__).resolve().parent.parent / path

    with path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            songs.append(
                {
                    "id": int(row["id"]),
                    "title": row["title"],
                    "artist": row["artist"],
                    "genre": row["genre"],
                    "mood": row["mood"],
                    "energy": float(row["energy"]),
                    "tempo_bpm": float(row["tempo_bpm"]),
                    "valence": float(row["valence"]),
                    "danceability": float(row["danceability"]),
                    "acousticness": float(row["acousticness"]),
                }
            )
    return songs

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """Score one song against user preferences and return score plus reasons."""
    score = 0.0
    reasons: List[str] = []

    favorite_genre = str(user_prefs.get("favorite_genre", "")).strip().lower()
    favorite_mood = str(user_prefs.get("favorite_mood", "")).strip().lower()
    target_energy = float(user_prefs.get("target_energy", 0.5))
    likes_acoustic = bool(user_prefs.get("likes_acoustic", False))

    song_genre = str(song.get("genre", "")).strip().lower()
    song_mood = str(song.get("mood", "")).strip().lower()
    song_energy = float(song.get("energy", 0.5))
    song_acousticness = float(song.get("acousticness", 0.0))

    # Sensitivity experiment (weight shift):
    # genre weight halved, energy weight doubled.
    if favorite_genre and song_genre == favorite_genre:
        score += 1.0
        reasons.append("genre match (+1.0)")

    if favorite_mood and song_mood == favorite_mood:
        score += 1.0
        reasons.append("mood match (+1.0)")

    energy_similarity = max(0.0, 1.0 - abs(song_energy - target_energy))
    energy_points = 4.0 * energy_similarity
    score += energy_points
    reasons.append(f"energy similarity (+{energy_points:.2f})")

    acoustic_match = (likes_acoustic and song_acousticness >= 0.5) or (
        (not likes_acoustic) and song_acousticness < 0.5
    )
    if acoustic_match:
        score += 0.5
        reasons.append("acoustic preference aligned (+0.5)")

    return score, reasons

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """Rank songs by score and return the top-k with explanations."""
    scored: List[Tuple[Dict, float, str]] = []
    for song in songs:
        score, reasons = score_song(user_prefs, song)
        explanation = "; ".join(reasons)
        scored.append((song, score, explanation))

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:k]
