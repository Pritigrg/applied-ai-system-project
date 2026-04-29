from typing import Dict

# Maps j-hartmann emotion labels to therapeutic music preferences (same format as MOOD_MAP).
# Counter-regulate negative states; reinforce positive ones.
EMOTION_TO_PREFS: Dict[str, Dict] = {
    "anger":   {"favorite_genre": "lofi",      "favorite_mood": "chill",      "target_energy": 0.4,  "likes_acoustic": True},
    "disgust": {"favorite_genre": "ambient",   "favorite_mood": "calm",       "target_energy": 0.2,  "likes_acoustic": True},
    "fear":    {"favorite_genre": "indie pop", "favorite_mood": "uplifted",   "target_energy": 0.65, "likes_acoustic": False},
    "joy":     {"favorite_genre": "pop",       "favorite_mood": "happy",      "target_energy": 0.8,  "likes_acoustic": False},
    "neutral": {"favorite_genre": "jazz",      "favorite_mood": "relaxed",    "target_energy": 0.35, "likes_acoustic": True},
    "sadness": {"favorite_genre": "classical", "favorite_mood": "reflective", "target_energy": 0.25, "likes_acoustic": True},
    "surprise":{"favorite_genre": "pop",       "favorite_mood": "festive",    "target_energy": 0.9,  "likes_acoustic": False},
}


def llm_classify_emotion(text: str, emotion_pipeline) -> Dict:
    """Classify emotion from text using a HuggingFace pipeline.

    Returns the standard user_prefs dict (4 keys for score_song) plus
    '_emotion_label' and '_emotion_confidence' for display. Extra keys
    are ignored by score_song() which only reads the 4 expected keys.
    """
    result = emotion_pipeline(text)
    # Normalize across all pipeline output shapes: {}, [{}], [[{}]]
    if isinstance(result, dict):
        top = result
    elif isinstance(result, list) and isinstance(result[0], list):
        top = result[0][0]
    else:
        top = result[0]
    label = top["label"].lower()
    confidence = top["score"]

    prefs = dict(EMOTION_TO_PREFS.get(label, _DEFAULT_PREFS))
    prefs["_emotion_label"] = label
    prefs["_emotion_confidence"] = confidence
    return prefs


# Maps mood keywords to therapeutic music preferences.
# Strategy: counter-regulate negative states (anxious → calm), reinforce positive ones.
MOOD_MAP = [
    (["anxious", "anxiety", "panic", "worried", "nervous", "stressed", "overwhelmed", "tense"],
     {"favorite_genre": "ambient",    "favorite_mood": "calm",        "target_energy": 0.2, "likes_acoustic": True}),

    (["sad", "sadness", "depressed", "lonely", "heartbroken", "down", "grief", "crying", "hopeless"],
     {"favorite_genre": "classical",  "favorite_mood": "reflective",  "target_energy": 0.25, "likes_acoustic": True}),

    (["angry", "anger", "furious", "frustrated", "irritated", "rage", "mad", "annoyed"],
     {"favorite_genre": "lofi",       "favorite_mood": "chill",       "target_energy": 0.4, "likes_acoustic": True}),

    (["tired", "exhausted", "drained", "sleepy", "fatigue", "burnt", "burnout"],
     {"favorite_genre": "ambient",    "favorite_mood": "calm",        "target_energy": 0.2, "likes_acoustic": True}),

    (["focused", "productive", "studying", "working", "concentrate", "deep work"],
     {"favorite_genre": "lofi",       "favorite_mood": "focused",     "target_energy": 0.4, "likes_acoustic": True}),

    (["happy", "excited", "joy", "great", "amazing", "good", "wonderful", "energized", "pumped"],
     {"favorite_genre": "pop",        "favorite_mood": "happy",       "target_energy": 0.8, "likes_acoustic": False}),

    (["nostalgic", "memories", "miss", "reminisce", "childhood", "bittersweet"],
     {"favorite_genre": "folk",       "favorite_mood": "nostalgic",   "target_energy": 0.4, "likes_acoustic": True}),

    (["festive", "party", "celebrate", "dancing", "dance", "hype"],
     {"favorite_genre": "pop",        "favorite_mood": "festive",     "target_energy": 0.9, "likes_acoustic": False}),

    (["moody", "pensive", "reflective", "contemplative", "thinking", "introspective"],
     {"favorite_genre": "synthwave",  "favorite_mood": "moody",       "target_energy": 0.5, "likes_acoustic": False}),

    (["relaxed", "peaceful", "content", "calm", "serene", "at ease", "chill"],
     {"favorite_genre": "jazz",       "favorite_mood": "relaxed",     "target_energy": 0.35, "likes_acoustic": True}),

    (["melancholic", "wistful", "blue", "gloomy", "somber"],
     {"favorite_genre": "blues",      "favorite_mood": "melancholic", "target_energy": 0.4, "likes_acoustic": True}),

    (["uplifted", "hopeful", "optimistic", "inspired", "motivated"],
     {"favorite_genre": "indie pop",  "favorite_mood": "uplifted",    "target_energy": 0.65, "likes_acoustic": False}),
]

_DEFAULT_PREFS = {"favorite_genre": "ambient", "favorite_mood": "calm", "target_energy": 0.3, "likes_acoustic": True}


def mood_to_prefs(text: str) -> Dict:
    """Map a free-text mood description to music therapy preferences."""
    lowered = text.lower()
    for keywords, prefs in MOOD_MAP:
        if any(kw in lowered for kw in keywords):
            return prefs
    return _DEFAULT_PREFS


def mood_label(prefs: Dict) -> str:
    """Return a short human-readable label for the mapped therapeutic intent."""
    mood = prefs.get("favorite_mood", "calm")
    genre = prefs.get("favorite_genre", "ambient")
    energy = prefs.get("target_energy", 0.3)
    if energy < 0.3:
        intensity = "gentle"
    elif energy < 0.6:
        intensity = "moderate"
    else:
        intensity = "upbeat"
    return f"{intensity} {mood} {genre}"
