import os
import requests
from typing import List, Dict, Optional

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def call_groq_chatbot(messages: List[Dict[str, str]]) -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "GROQ_API_KEY is not set."

    model_name = os.environ.get("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")

    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        payload = {"model": model_name, "messages": messages, "temperature": 0.3}
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        if not resp.ok:
            try:
                data = resp.json()
                err_msg = data.get("error", {}).get("message") or str(data)
                return f"Groq API error ({resp.status_code}): {err_msg}"
            except Exception:
                return f"Groq API HTTP error ({resp.status_code}): {resp.text}"

        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    except Exception as e:
        return f"Error calling Groq API: {e}"


def get_chatbot_reply(
    user_msg: str,
    last_prediction: Optional[int],
    last_probability: Optional[float],
) -> str:
    system_prompt = (
        "You are a helpful assistant inside a heart disease risk prediction app. "
        "Explain predictions in simple words, discuss risk factors and prevention, "
        "explain ML model features, and ALWAYS warn that this is not medical advice."
    )

    if last_prediction is not None and last_probability is not None:
        risk_text = "high" if last_prediction == 1 else "low"
        context_msg = (
            f"The last prediction was {risk_text} risk (class={last_prediction}) "
            f"with probability {last_probability:.2%}."
        )
    else:
        context_msg = "There is no recent model prediction."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": context_msg},
        {"role": "user", "content": user_msg},
    ]

    return call_groq_chatbot(messages)


@st.cache_data
def load_data(csv_path: str = "heart.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


@st.cache_resource
def train_model(df: pd.DataFrame):
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=200, random_state=42, max_depth=None, min_samples_split=2, min_samples_leaf=1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return model, acc, X.columns


def main():
    st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")

    if "show_chatbot" not in st.session_state:
        st.session_state.show_chatbot = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history: List[Dict[str, str]] = []
    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = None
    if "last_probability" not in st.session_state:
        st.session_state.last_probability = None

    st.title("❤️ Heart Disease Prediction Web App")
    st.write(
        "This app predicts heart disease risk using a machine learning model and an AI chatbot.\n"
        "This is for educational purposes only, not medical advice."
    )

    try:
        df = load_data("heart.csv")
    except FileNotFoundError:
        st.error("heart.csv not found.")
        return

    model, accuracy, feature_names = train_model(df)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Patient Information")

        age = st.number_input("Age", min_value=1, max_value=120, value=50, step=1)
        sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        cp = st.selectbox("Chest Pain (cp)", [0, 1, 2, 3])
        trestbps = st.number_input("Resting Blood Pressure", 80, 250, 130)
        chol = st.number_input("Cholesterol", 100, 600, 250)
        fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
        restecg = st.selectbox("Resting ECG", [0, 1, 2])
        thalach = st.number_input("Max Heart Rate", 60, 250, 150)
        exang = st.selectbox("Exercise-induced Angina", [0, 1])
        oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 10.0, 1.0, step=0.1)
        slope = st.selectbox("Slope of ST segment", [0, 1, 2])
        ca = st.number_input("Vessels Colored (ca)", 0, 3, 0)
        thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

        input_df = pd.DataFrame(
            [{
                "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
                "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
                "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
            }]
        )

        st.subheader("Input Summary")
        st.dataframe(input_df)

        if st.button("Predict"):
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]

            st.session_state.last_prediction = int(prediction)
            st.session_state.last_probability = float(probability)
            st.session_state.show_chatbot = True

            if prediction == 1:
                st.error(f"High risk of heart disease ({probability:.2%})")
            else:
                st.success(f"Low risk of heart disease ({probability:.2%})")

    with col2:
        st.header("Model Info")
        st.metric("Random Forest Accuracy", f"{accuracy:.2%}")


        st.subheader("Input Features")
        st.markdown(
            "- `age`: Age in years\n"
            "- `sex`: 0 = female, 1 = male\n"
            "- `cp`: Chest pain type (0–3)\n"
            "- `trestbps`: Resting blood pressure (mm Hg)\n"
            "- `chol`: Serum cholesterol (mg/dl)\n"
            "- `fbs`: Fasting blood sugar > 120 mg/dl (1 = yes, 0 = no)\n"
            "- `restecg`: Resting ECG results (0–2)\n"
            "- `thalach`: Maximum heart rate achieved\n"
            "- `exang`: Exercise-induced angina (1 = yes, 0 = no)\n"
            "- `oldpeak`: ST depression induced by exercise\n"
            "- `slope`: Slope of peak exercise ST segment (0–2)\n"
            "- `ca`: Number of major vessels (0–3)\n"
            "- `thal`: Thalassemia (0–3, encoded)\n"
        )

        with st.expander("Dataset Preview"):
            st.write(df.head())

    if st.session_state.show_chatbot:
        st.markdown("---")
        st.header("Chatbot")

        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['text']}")
            else:
                st.markdown(f"**Bot:** {msg['text']}")

        with st.form("chat_form", clear_on_submit=True):
            user_chat_input = st.text_input("Ask something about your result or heart health:", key="chat_input_form")
            submitted = st.form_submit_button("Send")

        if submitted and user_chat_input.strip():
            st.session_state.chat_history.append({"role": "user", "text": user_chat_input})
            bot_reply = get_chatbot_reply(
                user_chat_input,
                st.session_state.last_prediction,
                st.session_state.last_probability,
            )
            st.session_state.chat_history.append({"role": "bot", "text": bot_reply})


if __name__ == "__main__":
    main()
