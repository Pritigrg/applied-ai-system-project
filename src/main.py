"""
Command line runner for the Music Recommender Simulation.

Usage:
  python -m src.main                 # run hardcoded profile demo
  python -m src.main --interactive   # answer questions, get recommendations + live audio
"""

import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("vibematch.log"),
    ],
)

try:
    from recommender import load_songs, recommend_songs
except ModuleNotFoundError:
    from src.recommender import load_songs, recommend_songs

GENRES = [
    "lofi", "pop", "rock", "synthwave", "jazz", "ambient",
    "indie pop", "world", "metal", "blues", "classical", "reggae",
    "country", "folk", "drum and bass",
]
MOODS = [
    "chill", "happy", "intense", "moody", "focused", "relaxed", "festive",
    "aggressive", "melancholic", "calm", "uplifted", "euphoric", "nostalgic", "reflective",
]


def demo_mode(songs) -> None:
    print(f"Loaded songs: {len(songs)}")

    profiles = [
        (
            "High-Energy Pop",
            {
                "favorite_genre": "pop",
                "favorite_mood": "happy",
                "target_energy": 0.9,
                "likes_acoustic": False,
            },
        ),
        (
            "Chill Lofi",
            {
                "favorite_genre": "lofi",
                "favorite_mood": "chill",
                "target_energy": 0.4,
                "likes_acoustic": True,
            },
        ),
        (
            "Deep Intense Rock",
            {
                "favorite_genre": "rock",
                "favorite_mood": "intense",
                "target_energy": 0.85,
                "likes_acoustic": False,
            },
        ),
    ]

    for profile_name, user_prefs in profiles:
        recommendations = recommend_songs(user_prefs, songs, k=5)

        print(f"\n{profile_name}")
        print("=" * 60)
        for index, rec in enumerate(recommendations, start=1):
            song, score, explanation = rec
            reasons = [reason.strip() for reason in explanation.split(";") if reason.strip()]

            print(f"{index}. {song['title']} — {song['artist']}")
            print(f"   Final score: {score:.2f}")
            print("   Reasons:")
            for reason in reasons:
                print(f"   - {reason}")
            print("-" * 60)


def interactive_mode(songs) -> None:
    try:
        from agent import run_recommendation_with_audio
    except ModuleNotFoundError:
        from src.agent import run_recommendation_with_audio

    print("\nVibeMatch — Music Recommender + Audio Generator")
    print("Answer a few questions to get song picks and a live audio preview.")
    print("Type 'quit' at any prompt to exit.\n")

    while True:
        try:
            print(f"Genres: {', '.join(GENRES)}")
            genre = input("Favorite genre: ").strip().lower()
            if genre in ("quit", "exit", "q"):
                break
            if genre not in GENRES:
                print(f"  ('{genre}' not recognised — using 'ambient')")
                genre = "ambient"

            print(f"\nMoods: {', '.join(MOODS)}")
            mood = input("Current mood: ").strip().lower()
            if mood in ("quit", "exit", "q"):
                break
            if mood not in MOODS:
                print(f"  ('{mood}' not recognised — using 'calm')")
                mood = "calm"

            energy_raw = input("\nEnergy level 0.0 (very calm) to 1.0 (intense) [0.5]: ").strip()
            if energy_raw in ("quit", "exit", "q"):
                break
            try:
                energy = max(0.0, min(1.0, float(energy_raw)))
            except ValueError:
                energy = 0.5

            acoustic_raw = input("\nPrefer acoustic music? (y/n) [n]: ").strip().lower()
            if acoustic_raw in ("quit", "exit", "q"):
                break
            acoustic = acoustic_raw in ("y", "yes")

            user_prefs = {
                "favorite_genre": genre,
                "favorite_mood": mood,
                "target_energy": energy,
                "likes_acoustic": acoustic,
            }

            run_recommendation_with_audio(user_prefs, songs)

            again = input("\nTry another vibe? (y/n) [n]: ").strip().lower()
            if again not in ("y", "yes"):
                print("Goodbye!")
                break
            print()

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break


def main() -> None:
    songs = load_songs("data/songs.csv")

    if "--interactive" in sys.argv:
        interactive_mode(songs)
    else:
        demo_mode(songs)


if __name__ == "__main__":
    main()
