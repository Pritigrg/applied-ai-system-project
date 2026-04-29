# Reflection: Profile Pair Comparisons

I compared five profiles:
- High-Energy Pop
- Chill Lofi
- Deep Intense Rock
- Conflict: High Energy + Sad
- Edge: Unknown Genre

Below is one plain-language comparison for each profile pair.

## Pairwise Comments

1. High-Energy Pop vs Chill Lofi
High-Energy Pop pushes upbeat, fast songs, while Chill Lofi shifts toward calmer tracks. This makes sense because the target energy is much higher for the pop profile.

2. High-Energy Pop vs Deep Intense Rock
Both profiles share high energy, so some energetic songs overlap. The difference is that rock/intense preferences pull in heavier tracks, while pop/happy keeps brighter songs near the top.

3. High-Energy Pop vs Conflict: High Energy + Sad
These two are similar on energy, so some top songs overlap, including Gym Hero. The mood change to sad does not always dominate because energy still contributes a lot of points.

4. High-Energy Pop vs Edge: Unknown Genre
When the genre is unknown, the model cannot award genre-match points, so the list depends more on mood and energy. This is why some expected pop songs drop and more general energy-matched songs appear.

5. Chill Lofi vs Deep Intense Rock
This pair shows the clearest contrast: Chill Lofi favors low-energy, softer songs; Deep Intense Rock favors high-energy, aggressive tracks. The output split makes sense because the target energy and mood are very different.

6. Chill Lofi vs Conflict: High Energy + Sad
Chill Lofi stays calm and acoustic, while High Energy + Sad still pushes energetic songs upward. This shows how strong the energy feature is in separating outcomes.

7. Chill Lofi vs Edge: Unknown Genre
Both can include chill songs, but Unknown Genre loses genre points and becomes less focused. Chill Lofi feels more stable because lofi and chill both align with the dataset labels.

8. Deep Intense Rock vs Conflict: High Energy + Sad
Both are high energy, so energetic songs still rank well in both profiles. The difference is that rock/intense rewards songs with matching genre/mood, while high-energy sad can still look "gym-like" if energy is closest.

9. Deep Intense Rock vs Edge: Unknown Genre
Unknown Genre cannot use genre matching, so even if rock songs are available, they are not rewarded for genre identity. Deep Intense Rock gives clearer identity-based recommendations.

10. Conflict: High Energy + Sad vs Edge: Unknown Genre
Both profiles can produce surprising results because one has conflicting mood-energy goals and the other has no valid genre label. In both cases, energy similarity often becomes the tie-breaker, which is why very energetic tracks can still rise to the top.

## Plain-Language Takeaway
The model is doing what we told it to do: reward close energy matches strongly. That is why Gym Hero keeps showing up for users who say they want Happy Pop. It is not because the model "understands" gym music better; it is because Gym Hero has very high energy and gets many points from that part of the formula.

## Limitations, Bias, and Reflection

1) What are the limitations or biases in your system?

**Energy dominates scoring.** Energy has weight 4.0 out of a maximum 6.5 (62%). A song with the right energy but the wrong genre will almost always outrank a genre-perfect song with slightly different energy. Genre and mood each award only 1.0 point, so they rarely decide the final ranking.

**Small catalog.** The catalog has 22 songs. If a user's detected emotion maps to "classical" (e.g. sadness) but no classical songs exist in the catalog, the system silently picks the closest energy match with low confidence (~62%). There is no warning that the genre preference went unmet.

**Hard cutoffs.** Acoustic preference uses a cutoff at 0.5 acousticness — a song at 0.49 gets no credit even if the user asked for acoustic music. Genre and mood require exact text matches, so a catalog label mismatch silently loses the point.

**Emotion classifier is English-only and trained on formal text.** The `j-hartmann/emotion-english-distilroberta-base` model was fine-tuned on written datasets, not transcribed speech. Casual, short, or heavily accented phrases may be misclassified. "Meh" or "I dunno, just blah" are likely to fall through to `neutral` regardless of actual feeling.

**Whisper errors cascade.** If Whisper mishears a word, the emotion classifier receives wrong input and the entire downstream recommendation is off. There is no re-try or confidence check between the two AI stages.

**Seven emotion buckets is coarse.** Human emotion is not seven categories. A user who feels "nostalgic but content" gets mapped to whichever single label scores highest, which may not reflect the nuance they expressed.

---

2) Could your AI be misused, and how would you prevent that?

The voice input captures a raw audio recording. If deployed in a shared or public environment, this creates a privacy risk — someone else's voice could be recorded without consent. The current implementation processes audio only in memory and never writes it to disk, which limits but does not eliminate this risk.

The emotion classifier could be deliberately probed. A user who knows the seven label-to-preference mappings could speak a phrase designed to trigger a specific emotion label and get a specific genre back, bypassing the intended therapeutic logic. The `EMOTION_TO_PREFS` table is readable in `src/therapy.py`, so the mappings are not secret.

The therapeutic framing ("music therapy") could create a false sense of clinical support. The system has no clinical training and makes no medical claim, but a vulnerable user might interpret the recommendations as professional advice. A disclaimer in the UI would address this.

---

3) What surprised you while testing your AI's reliability?

Two things stood out.

First, an unknown or unmatched genre did not crash — it silently returned results with lower confidence (~62% vs ~95% for a matched profile). The system degrades gracefully, but you would never know the match was weak without the confidence percentage. That is why `confidence_pct()` matters: it makes the quality of each recommendation visible rather than hiding degradation behind a confident-looking list.

Second, the emotion classifier's output varied more than expected with phrasing. "I'm sad" and "I've been feeling really low and lonely all week" both get classified as `sadness`, but with very different confidence scores. Short, vague sentences consistently produced lower confidence and sometimes landed on `neutral` when the intended emotion was clear to a human reader. This suggests the UI should encourage users to speak in full sentences rather than single words.

---

4) Describe your collaboration with AI during this project. Identify one instance when the AI gave a helpful suggestion and one instance where its suggestion was flawed or incorrect.

I used Claude Code throughout the project — for designing the pipeline architecture, writing tests, debugging the HuggingFace pipeline output shape normalization in `therapy.py`, and iterating on the Streamlit UI layout.

*Helpful suggestion:* Based on my requirement, claude was able to suggest me good models for transcribing text and emotion recognition model.

*Flawed suggestion:* An early design used a cloud LLM (Claude API) to parse free-text mood input via a tool-calling loop. That approach required an API key, making the project not reproducable without a paid account.
