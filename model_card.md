# Model Card: EmotionLift — Emotion-Based Music Recommender

---

## 1. Model Name

**EmotionLift v2** — a voice-driven music recommendation system that detects emotion from speech and recommends songs matched to how the user feels.

---

## 2. Intended Use

EmotionLift is designed to recommend music based on how a user is feeling, detected automatically from their voice. It is intended for:

- Users who want music matched to their current emotional state without filling in a form
- How speech recognition and emotion classification can be combined with a rule-based recommender
- Demonstrating how two specialized AI models can be composed into a transparent, auditable pipeline

The system assumes the user speaks in English. It is for educational exploration, not clinical therapy.

---

## 3. How the Model Works

The system is a four-stage pipeline:

**Stage 1 — Speech to Text (`openai/whisper-base`)**
The user speaks into their browser microphone. Whisper converts the audio to a plain-text sentence. It handles casual speech, background noise, and varied accents.

**Stage 2 — Emotion Classification (`j-hartmann/emotion-english-distilroberta-base`)**
The transcribed sentence is passed to an emotion classifier — a DistilRoBERTa model fine-tuned on six emotion-labelled datasets. It assigns one of seven labels: `anger`, `disgust`, `fear`, `joy`, `neutral`, `sadness`, `surprise`, and returns a confidence score (0–1).

**Stage 3 — Emotion-to-Preference Mapping (`EMOTION_TO_PREFS` in `therapy.py`)**
Each emotion label maps to a fixed set of music preferences encoded in a lookup table. The logic is: calm down negative states (e.g. anger → lofi/chill, sadness → classical/reflective) and match positive ones (joy → pop/happy). This table is fully readable in `src/therapy.py`.

**Stage 4 — Rule-Based Recommender (`src/recommender.py`)**
Each song in the 22-song catalog is scored using four weighted features:
- Genre match: +1.0
- Mood match: +1.0
- Energy closeness: up to +4.0
- Acoustic alignment: +0.5 (max score: 6.5)

Songs are ranked by score and the top 5 are returned with a plain-English explanation and a confidence percentage. Real MP3 files from `data/media/` play inline in the Streamlit UI.

---

## 4. Data

The song catalog (`data/songs.csv`) contains **22 songs** spanning genres including pop, rock, folk, jazz, reggae, country, and blues. Moods include happy, uplifted, reflective, festive, chill, nostalgic, calm, relaxed, and moody.

Each song has seven numeric and text attributes: `genre`, `mood`, `energy` (0–1), `tempo_bpm`, `valence`, `danceability`, and `acousticness` (0–1).

The catalog was expanded from 18 songs (v1) to 22 songs to improve coverage. Notable gaps remain: there are no classical songs despite `sadness` mapping to classical preferences, and several genres (jazz, reggae, country) have only one representative song.

---

## 5. Strengths

- **Transparent scoring**: every recommendation shows exactly which features contributed and by how much — no black box
- **Graceful degradation**: an unmatched genre lowers confidence (to ~62%) without crashing
- **Two-layer confidence**: the UI shows both the emotion classifier's confidence (e.g. `fear · 87%`) and the recommender's match percentage per song
- **Natural input**: speaking a sentence gives the emotion classifier far richer signal than a single keyword or dropdown
- **Fully local**: no API key required after a one-time model download

---

## 6. Limitations and Bias

**Energy dominates.** Energy carries weight 4.0 out of 6.5 (62%). A song with the right energy but the wrong genre almost always outranks a genre-perfect song with slightly different energy.

**Small catalog.** With 22 songs, niche genre preferences (e.g. "classical" for sadness) often go unmet. The system silently picks the next best match with lower confidence.

**Emotion classifier trained on written text.** The model was fine-tuned on formal written datasets, not transcribed speech. Short or vague phrases ("I dunno, meh") consistently fall through to `neutral`. Accented or non-standard English may be misclassified.

**Whisper errors cascade.** A transcription error feeds bad input to the emotion classifier, shifting the entire recommendation with no recovery path.

**Seven emotion buckets is coarse.** Human emotion is more nuanced than seven fixed categories. "Nostalgic but content" gets flattened to whichever single label scores highest.

**Classroom scope.** This is a classroom exploration project, not a clinical or production system. It makes no medical claims and has not been validated beyond manual testing.

---

## 7. Evaluation

**Automated tests:** 11 tests pass in 0.25 seconds (`python -m pytest tests/ -v`). Tests cover the scoring engine, ranking order, confidence percentage calculation, vibe summarization, and graceful degradation for unknown genres. All model and audio dependencies are mocked.

**Confidence scoring:** matched emotion profiles (e.g. joy → pop/happy) produce top scores of 6.18/6.5 (95% confidence). Unmatched genre preferences (sadness → classical, but no classical songs in catalog) drop to ~4.92/6.5 (62%).

**Logging:** `src/agent.py` writes timestamped session logs capturing genre, mood, energy, top recommendation, and confidence on every run.

**Human evaluation:** after each session, the listener checks whether the detected emotion label matches what they said, and whether the top songs feel like a good match for that emotion.

**Gap:** `src/therapy.py` and `src/voice_input.py` have no unit tests yet — these are the next tests to write.

---

## 8. Future Work

- Add unit tests for `therapy.py` — mock the HuggingFace pipeline and assert each emotion label maps to the correct `user_prefs` dict
- Expand the catalog to include classical songs so sadness recommendations match the genre target
- Add soft matching for near-miss genres and moods (e.g. "lo-fi" matches "lofi")
- Replace the hard acoustic cutoff (0.5) with a continuous score
- Add image-based emotion input (detect emotion from a photo) — the recommender and mapping layer would need no changes
- A/B test voice input vs. a keyword form to measure whether emotion detection actually improves recommendation quality

---

## 9. AI Collaboration

I used Claude Code throughout the project — for designing the pipeline architecture, wiring together Whisper and the emotion classifier, building the Streamlit UI, writing tests, and debugging HuggingFace pipeline output shape issues in `therapy.py`.

*Helpful suggestion:* Based on my requirements, Claude suggested good models for transcribing speech (`openai/whisper-base`) and for emotion recognition (`j-hartmann/emotion-english-distilroberta-base`), explaining the trade-offs between model size and accuracy for a local, no-API-key setup.

*Flawed suggestion:* An early design used a cloud LLM (Claude API) to parse free-text mood input via a tool-calling loop. That approach required an API key, making the project not reproducible without a paid account. I replaced it with local Whisper + local HuggingFace emotion classifier — no API key needed and a cleaner fit for the assignment's specialized model requirement.

