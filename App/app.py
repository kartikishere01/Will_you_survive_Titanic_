import streamlit as st
import pandas as pd
import joblib
import os
import random

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Titanic Survival Fun App",
    page_icon="🚢",
    layout="wide"
)

# ---------- LOAD MODEL ----------
MODEL_PATH = "titanic_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file '{MODEL_PATH}' not found in this folder.")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ---------- TITANIC FUN FACTS ----------
FUN_FACTS = [
    "There were only 20 lifeboats on Titanic for about 2,200 people.",
    "The ship had a gym, swimming pool, Turkish bath, and even a squash court.",
    "Titanic’s top speed was about 23 knots (around 42 km/h).",
    "The famous 'Unsinkable' claim was mostly media hype, not an official slogan.",
    "About 60% of first-class passengers survived, but only ~25% of third-class."
]

# ---------- HEADER ----------
st.markdown(
    """
    <h1 style='text-align: center;'>
        🚢 Titanic Survival Predictor 🎲
    </h1>
    <p style='text-align: center; font-size: 16px;'>
        Answer a few questions and see if <b>you</b> might have survived the Titanic night...
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ---------- SIDEBAR ----------
st.sidebar.title("About this app")
st.sidebar.write(
    """
    This app uses a simple Machine Learning model trained on the classic Titanic dataset.
    
    It considers:
    - Ticket class (as income bracket)
    - Gender
    - Age
    
    Then it predicts your chance of survival.
    """
)

st.sidebar.subheader("Random Titanic Fact")
st.sidebar.info(random.choice(FUN_FACTS))

# ---------- MAIN LAYOUT ----------
left_col, right_col = st.columns([1.1, 1])

with left_col:
    st.subheader("🧾 Fill your details")

    name = st.text_input("Your Name", placeholder="Enter your name (optional)")

    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)

    age = st.slider("Age", min_value=1, max_value=80, value=25)

    income_bracket = st.selectbox(
        "Income Bracket (approx.)",
        ["Low (3rd Class)", "Middle (2nd Class)", "High (1st Class)"]
    )

    # Map income bracket to Pclass
    if "High" in income_bracket:
        pclass = 1
    elif "Middle" in income_bracket:
        pclass = 2
    else:
        pclass = 3

    # Map gender to numeric
    sex_val = 1 if gender == "Male" else 0

    predict_button = st.button("🔮 Will I Survive?")

with right_col:
    st.subheader("📊 Result")

    if predict_button:
        # Prepare input
        input_df = pd.DataFrame(
            [[pclass, sex_val, age]],
            columns=['Pclass', 'Sex', 'Age']
        )

        try:
            pred = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0][1]  # probability of survival
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

        person = name.strip() if name.strip() != "" else "You"

        # Survival meter
        st.write("### 🎚 Survival Chance Meter")
        st.progress(min(max(proba, 0.0), 1.0))
        st.write(f"**Estimated survival chance:** `{proba * 100:.1f}%`")

        # Text result
        if pred == 1:
            st.success(f"✅ {person} would likely **SURVIVE** the Titanic.")
            st.markdown(
                "> You manage to find a spot on a lifeboat after a tense wait on deck. "
                "Cold wind, loud cries, but you make it out alive. 🛟"
            )
        else:
            st.error(f"❌ {person} would likely **NOT SURVIVE** the Titanic.")
            st.markdown(
                "> You stay back in the freezing chaos, "
                "helping others reach the boats as the ship tilts dangerously... ❄️"
            )

        # Show a nice summary "ticket"
        st.write("### 🎫 Your Titanic Ticket Summary")
        summary_df = pd.DataFrame({
            "Field": ["Name", "Gender", "Age", "Income Bracket (Pclass)"],
            "Value": [
                person,
                gender,
                age,
                f"{income_bracket}  → Pclass {pclass}"
            ]
        })
        st.table(summary_df)

    else:
        st.info("Fill the details on the left and click **'Will I Survive?'** to see your prediction.")
