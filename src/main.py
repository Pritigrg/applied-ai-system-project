"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from recommender import load_songs, recommend_songs


def main() -> None:
    songs = load_songs("data/songs.csv") 
    print(f"Loaded songs: {len(songs)}")

    # Specific taste profile: calm late-night focus listener
    user_prefs = {
        "favorite_genre": "lofi",
        "favorite_mood": "chill",
        "target_energy": 0.4,
        "likes_acoustic": True,
    }

    recommendations = recommend_songs(user_prefs, songs, k=5)

    print("\nTop recommendations\n")
    print("=" * 60)
    for index, rec in enumerate(recommendations, start=1):
        # recommend_songs returns: (song, score, explanation)
        song, score, explanation = rec
        reasons = [reason.strip() for reason in explanation.split(";") if reason.strip()]

        print(f"{index}. {song['title']} — {song['artist']}")
        print(f"   Final score: {score:.2f}")
        print("   Reasons:")
        for reason in reasons:
            print(f"   - {reason}")
        print("-" * 60)


if __name__ == "__main__":
    main()
