# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name  

Give your model a short, descriptive name.  
Example: **VibeFinder 1.0**  

VibeMatch Classroom Recommender v1.

---

## 2. Intended Use  

Describe what your recommender is designed to do and who it is for. 

Prompts:  

- What kind of recommendations does it generate  
- What assumptions does it make about the user  
- Is this for real users or classroom exploration  

This recommender suggests songs from a small class dataset.
It matches user taste to song features.
It assumes users know their genre, mood, and energy preference.
This is for classroom exploration, not real production use.

---

## 3. How the Model Works  

Explain your scoring approach in simple language.  

Prompts:  

- What features of each song are used (genre, energy, mood, etc.)  
- What user preferences are considered  
- How does the model turn those into a score  
- What changes did you make from the starter logic  

Avoid code here. Pretend you are explaining the idea to a friend who does not program.

Each song has genre, mood, energy, and acousticness.
The user profile has favorite genre, favorite mood, target energy, and acoustic preference.
The model adds points for genre and mood matches.
It adds more points when song energy is close to target energy.
It adds a small bonus if acoustic preference aligns.
Then songs are sorted by score and top songs are returned.

I changed the starter scoring.
Genre weight is lower now.
Energy weight is higher now.
So energy affects the final ranking more.

---

## 4. Data  

Describe the dataset the model uses.  

Prompts:  

- How many songs are in the catalog  
- What genres or moods are represented  
- Did you add or remove data  
- Are there parts of musical taste missing in the dataset  

The catalog has 18 songs.
It includes lofi, pop, rock, jazz, classical, reggae, and other genres.
It includes moods like chill, happy, intense, reflective, and calm.
I did not add or remove songs.
Some music tastes are still missing because the dataset is small.
Many genres appear only once.

---

## 5. Strengths  

Where does your system seem to work well  

Prompts:  

- User types for which it gives reasonable results  
- Any patterns you think your scoring captures correctly  
- Cases where the recommendations matched your intuition  

The model works well for clear user profiles.
It performs well for High-Energy Pop and Chill Lofi profiles.
Energy and mood matching often feel correct.
The results are easy to explain because the score is transparent.

---

## 6. Limitations and Bias 

Where the system struggles or behaves unfairly. 

Prompts:  

- Features it does not consider  
- Genres or moods that are underrepresented  
- Cases where the system overfits to one preference  
- Ways the scoring might unintentionally favor some users  

This recommender is simple and transparent, but it has important limits.
It does not consider lyrics, artist history, or listening context.
Some genres and moods are underrepresented because the dataset is small.
Energy has high weight, so recommendations can become repetitive.
Genre and mood need exact text matches, which can hurt users with uncommon labels.
The acoustic rule uses a hard cutoff and misses nuance near the threshold.

---

## 7. Evaluation  
How you checked whether the recommender behaved as expected.

Prompts:

Which user profiles you tested
What you looked for in the recommendations
What surprised you
Any simple tests or comparisons you ran
No need for numeric metrics unless you created some.

I tested High-Energy Pop, Chill Lofi, and Deep Intense Rock profiles.
I checked whether the top songs matched each profile's vibe.
I also checked if the explanation text matched scoring behavior.
I compared profile outputs to see overlap and differences.
The biggest surprise was how strongly energy drives ranking after the weight change.
I also used unit tests for ranking order and non-empty explanations.

---

## 8. Future Work  

Ideas for how you would improve the model next.  

Prompts:  

- Additional features or preferences  
- Better ways to explain recommendations  
- Improving diversity among the top results  
- Handling more complex user tastes  

I would add more songs to improve coverage.
I would add soft matching for similar genres and moods.
I would include more features like tempo, danceability, and valence.
I would improve diversity so top results are less repetitive.
I would make explanations even clearer and more personal.

---

## 9. Personal Reflection  

A few sentences about your experience.  

Prompts:  

- What you learned about recommender systems  
- Something unexpected or interesting you discovered  
- How this changed the way you think about music recommendation apps  

I learned that small weight changes can shift recommendations a lot.
I learned that transparent scoring makes debugging easier.
I was surprised by how quickly energy can dominate results.
This project made me think more about fairness and diversity in real apps.
