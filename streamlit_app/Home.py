import streamlit as st

st.set_page_config(
    page_title="Pidgin Sentiment Dashboard",
    page_icon="🇳🇬",
    layout="wide",
)

st.title("Nigerian Pidgin Sentiment & Emotion Dashboard")
st.caption("Built on the WAZOBIALABS dataset")

st.markdown("""
This dashboard analyses sentiment and emotion in Nigerian Pidgin text — a high-context 
language where the same words can carry completely different meaning depending on tone, 
register, and emotional context.

Use the sidebar to navigate between pages.
""")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Analysis Pages")
    st.markdown("""
- **Dataset Overview** — distributions and dataset characteristics: sentiment breakdown, 
  emotion frequency, speaker demographics, register, topic domain, and intensity spread
- **Sentiment & Emotion Analysis** — cross-tabulations, sarcasm vs sentiment, 
  register vs sentiment, and the 28 context-variant pairs where the same text 
  carries opposite sentiment depending on emotional register
- **Model Performance** — accuracy comparison across experiments (text-only baseline 
  ~51% vs emotion-aware fine-tuned model 96.4%), confusion matrices, per-class F1, 
  and the data leakage investigation
""")

with col2:
    st.subheader("🤖 Agent Page")
    st.markdown("""
The agent page lets you interactively analyse new Nigerian Pidgin text. Here's how it works:

1. **Type your pidgin text** in the chat input box at the bottom
2. **The agent guesses the emotion** using Claude (Anthropic's LLM), choosing from the 16 emotion categories below
3. **Confirm or correct the emotion** using the dropdown — this is the key step, since 
   sentiment in Pidgin depends heavily on emotional register
4. **Click "Confirm and analyse sentiment"** — the fine-tuned model predicts sentiment 
   (positive / negative / neutral) using both the text and confirmed emotion as input
5. **The result shows** predicted sentiment and confidence score
""")

st.divider()

st.subheader("🎭 The 16 Emotion Categories")
st.caption("Designed by Stephanie Nkemjika Okoye to capture emotional registers specific to Nigerian Pidgin")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
**anger**  
Frustration or rage

**betrayal**  
Feeling let down or deceived

**celebration**  
Joy over an achievement or win

**contempt**  
Cold downward dismissal, distinct from anger
""")

with col2:
    st.markdown("""
**craving**  
Intense desire or longing

**forming**  
Performing unbothered on purpose

**hustle_energy**  
Motivated, grinding energy

**hustle_fatigue**  
So worn down by grinding you can't react
""")

with col3:
    st.markdown("""
**joy**  
Genuine happiness or delight

**market_energy**  
Sharp transactional energy of Nigerian negotiation

**neutral**  
No strong emotional charge

**prayer_gratitude**  
The testimony register — thankfulness and praise
""")

with col4:
    st.markdown("""
**pride**  
Satisfaction in oneself or others

**sarcasm**  
Saying the opposite of what you mean

**shock**  
Surprise or disbelief

**suspicion**  
Wariness or distrust
""")

st.divider()

st.info("""
**Why emotion matters for Pidgin sentiment:** Text-only sentiment classification 
caps at ~51% accuracy because the same surface text carries opposite sentiment 
depending on context. Providing the emotion category lifts accuracy to 96.4% — 
quantifying how much meaning in Pidgin lives outside the words themselves.
""")
