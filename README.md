# Nigerian Pidgin Sentiment & Emotion Analysis 

An end-to-end NLP pipeline for analysing sentiment and emotion in Nigerian Pidgin text, a language where tone, sarcasm, and cultural context carry as much meaning as the words themselves.

---

## Project Overview

Nigerian Pidgin is a high-context language where the same surface text can express opposite sentiments depending on tone, emotional register, and intent. The sentence "You do well", said in pride versus said in sarcasm, carries completely different meaning. This project investigates how much of that sentiment signal is recoverable from text alone versus emotional context, builds fine-tuned classification models that explicitly leverage emotion as an input feature alongside text, and wraps the best-performing approach in an AI agent that first identifies the emotional context of new Pidgin text before predicting its sentiment, reflecting the reality that in Pidgin, emotion is not a byproduct of sentiment but a prerequisite for understanding it.

---

## Motivation

This project began with a LinkedIn post by Stephanie Nkemjika Okoye, the dataset author, describing her experience annotating Nigerian Pidgin text find post [here](https://www.linkedin.com/posts/sokoye_nigerianpidgin-africannlp-lowresourcenlp-share-7459283130651246592-2LgC/?utm_source=social_share_send&utm_medium=member_desktop_web&rcm=ACoAADfXXRoBKaF9-aJ0K2D1TJI7usOMd8BjKAQ):

> "Most NLP datasets for African languages use positive / negative / neutral. Three buckets. For English, that's workable. For Nigerian Pidgin, it's almost meaningless. When someone says 'I no fit shout' -- are they angry? Exhausted? Indifferent? The answer is none of those. They're most likely in hustle_fatigue. A state of being so worn down by grinding that you can't even summon the energy to react. That's a specific Nigerian emotional register that doesn't map to any standard sentiment label."

She went on to describe building a 16-category emotion taxonomy from scratch, grounded in how Nigerians actually speak, including categories like `forming` (performing unbothered on purpose), `market_energy` (the sharp transactional energy of Nigerian negotiation), `prayer_gratitude` (the testimony register), and `contempt` (a cold downward dismissal distinct from anger). The dataset included 39 sarcasm pairs with sincere twins: identical surface text, opposite sentiment, different emotional register, a deliberate design choice to capture the nuance that standard positive/negative/neutral annotation doesn't capture.



---

## Dataset

- **Source**: `WAZOBIALABS/nigerian-pidgin-voice-text` (500 rows, training) and `WAZOBIALABS/nigerian-pidgin-eval` (253 rows, evaluation) via Hugging Face Hub
- **Columns**: `pidgin_text`, `sentiment`, `emotion_category`, `emotion_secondary`, `register`, `sarcasm_flag`, `prosody_match`, `intensity`, `speaker_gender`, `topic_domain`, `source_type`, `annotator_notes`, `date_added`
- **Sentiment classes**: positive, negative, neutral
- **Emotion classes**: 16 fine-grained categories (anger, betrayal, celebration, contempt, craving, forming, hustle_energy, hustle_fatigue, joy, market_energy, neutral, prayer_gratitude, pride, sarcasm, shock, suspicion)


## AI Agent

An AI agent was built to make the context-aware sentiment pipeline interactive, handling the problem that `emotion_category` is not available for new, unannotated text at inference time.

**Agent flow:**
1. User submits raw Pidgin text
2. Agent autonomously calls an LLM-based emotion guesser tool, reasoning over the result
3. The guessed emotion is shown to the user with a dropdown to confirm or correct it
4. Agent continues (preserving its reasoning memory via `reset=False`) and calls the sentiment classifier tool with the confirmed emotion
5. Final sentiment and confidence are reported

**Why LLM-based emotion guessing**: the fine-tuned 16-class emotion classifier (~33%) was not reliable enough for autonomous use. The LLM, constrained to choose from the 16 valid categories, produces more useful guesses given its broader multilingual training, even without explicit fine-tuning on this dataset.

**Web compatibility**: the agent uses a two-phase `CodeAgent` run (smolagents) split at the user confirmation step, avoiding the terminal-only `UserInputTool` blocking `input()` call. The agent is still genuinely agentic (tool-calling, autonomous reasoning) at each phase, just structured to fit a browser request/response model.

---

## Setup

### Requirements
```bash
pip install transformers torch datasets scikit-learn pandas smolagents litellm anthropic streamlit plotly streamlit-extras huggingface_hub
```

### Environment variables
Create a `.env` file at the project root:
```
HUGGINGFACE_KEY=hf_xxxxxxxxxxxx
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxx
```

### Run order

**1. Pre-compute analysis data** (required before running the dashboard):
```bash
python -m analysis.compute_analysis
```

**2. Run the dashboard:**
```bash
python -m streamlit run streamlit_app/Home.py
```

**3. Run the terminal agent demo** (optional, separate from the dashboard):
```bash
python -m agent.pidgin_agent
```

---

## Technologies

Python · HuggingFace Transformers · smolagents · LiteLLM · Anthropic Claude API · scikit-learn · Plotly · Streamlit · streamlit-extras

---

## Credits & Citations

### Datasets

**Training and evaluation data** created by Stephanie Nkemjika Okoye / Wazobia Labs. Published under **CC-BY-4.0**, free to use for research, academic, and commercial purposes with attribution.

- Training set: [`WAZOBIALABS/nigerian-pidgin-voice-text`](https://huggingface.co/datasets/WAZOBIALABS/nigerian-pidgin-voice-text)
- Evaluation set: [`WAZOBIALABS/nigerian-pidgin-eval`](https://huggingface.co/datasets/WAZOBIALABS/nigerian-pidgin-eval)
- Contact: wazobialabs@gmail.com

```bibtex
@dataset{wazobia_labs_pidgin_2026,
  author    = {Okoye, Stephanie Nkemjika},
  title     = {Wazobia Labs Nigerian Pidgin Emotion and Sentiment Dataset},
  year      = {2026},
  version   = {0.8.0},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/datasets/WAZOBIALABS/nigerian-pidgin-voice-text},
  license   = {CC-BY-4.0},
  note      = {First commercially licensed Nigerian Pidgin emotion dataset with 16-category cultural taxonomy}
}
```

### Pre-trained Models

**`Davlan/naija-twitter-sentiment-afriberta-large`** used as the sentiment classification baseline and fine-tuning starting point. Trained on the NaijaSenti corpus by Shamsuddeen Hassan Muhammad, David Ifeoluwa Adelani, Sebastian Ruder, and colleagues. If you use this model, please cite:

```bibtex
@inproceedings{muhammad-etal-2022-naijasenti,
    title = "{N}aija{S}enti: A {N}igerian {T}witter Sentiment Corpus for Multilingual Sentiment Analysis",
    author = "Muhammad, Shamsuddeen Hassan and Adelani, David Ifeoluwa and Ruder, Sebastian
              and Ahmad, Ibrahim Sa{'}id and Abdulmumin, Idris and Bello, Bello Shehu
              and Choudhury, Monojit and Emezue, Chris Chinenye and Abdullahi, Saheed Salahudeen
              and Aremu, Anuoluwapo and Jorge, Al{\'\i}pio and Brazdil, Pavel",
    booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
    month = jun,
    year = "2022",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2022.lrec-1.63",
    pages = "590--602",
}
```

**`Davlan/afro-xlmr-base`** used as the base encoder for emotion classification fine-tuning. Created by Jesujoba O. Alabi, David Ifeoluwa Adelani, Marius Mosbach, and Dietrich Klakow via multilingual adaptive fine-tuning on 17 African languages. If you use this model, please cite:

```bibtex
@inproceedings{alabi-etal-2022-adapting,
    title = "Adapting Pre-trained Language Models to {A}frican Languages via Multilingual Adaptive Fine-Tuning",
    author = "Alabi, Jesujoba O. and Adelani, David Ifeoluwa and Mosbach, Marius and Klakow, Dietrich",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2022.coling-1.382",
    pages = "4336--4349",
}
```

### LLM Agent

The emotion-guessing component of the agent uses **Claude** (Anthropic) via the Anthropic API and LiteLLM. Claude is a product of [Anthropic](https://www.anthropic.com).
