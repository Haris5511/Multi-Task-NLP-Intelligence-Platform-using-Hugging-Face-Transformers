# app.py

import streamlit as st

from sentiment import predict_sentiment
from ner import extract_entities
from summarization import summarize_text
from translation import translate_text
from question_answering import answer_question
from generation import generate_text
from zero_shot import classify_text


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AI NLP Studio",
    page_icon="🤖",
    layout="wide"
)


# ---------------- CUSTOM CSS ----------------

st.markdown(
    """
    <style>

    .main-title {
        font-size:40px;
        font-weight:700;
        text-align:center;
        color:#4A90E2;
    }

    .sub-title {
        text-align:center;
        font-size:18px;
        color:gray;
    }

    .result-box {
        padding:20px;
        border-radius:15px;
        background:#f5f7fb;
        border:1px solid #ddd;
        margin-top:20px;
    }

    .stButton>button {
        width:100%;
        height:50px;
        border-radius:12px;
        font-size:18px;
        font-weight:bold;
    }

    </style>
    """,
    unsafe_allow_html=True
)



# ---------------- HEADER ----------------


st.markdown(
    '<div class="main-title">🤖 AI Multi-Task NLP Studio</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Powered by Hugging Face Transformers</div>',
    unsafe_allow_html=True
)


st.write("")


# ---------------- SIDEBAR ----------------


with st.sidebar:

    st.header("⚙️ NLP Tasks")

    task = st.selectbox(
        "Choose Task",
        [
            "Sentiment Analysis",
            "Named Entity Recognition",
            "Text Summarization",
            "Translation",
            "Question Answering",
            "Text Generation",
            "Zero-Shot Classification"
        ]
    )


    st.divider()

    st.info(
        """
        This app supports multiple NLP
        tasks using Hugging Face
        Transformer models.
        """
    )



# ---------------- INPUT ----------------


st.subheader("📝 Input Text")


text = st.text_area(
    "Enter your text here",
    height=200,
    placeholder="Write or paste text..."
)



# ---------------- EXTRA INPUTS ----------------


labels = ""

question = ""


if task == "Zero-Shot Classification":

    st.subheader("🏷️ Categories")

    labels = st.text_input(
        "Enter labels separated by comma",
        "Technology, Sports, Politics, Business"
    )


if task == "Question Answering":

    st.subheader("❓ Question")

    question = st.text_input(
        "Ask your question"
    )



# ---------------- BUTTON ----------------


run = st.button(
    "🚀 Run Model"
)



# ---------------- MODEL EXECUTION ----------------


if run:


    if text.strip():


        with st.spinner("🤖 AI Model is working..."):



            # Sentiment

            if task == "Sentiment Analysis":


                result = predict_sentiment(text)


                st.subheader("😊 Sentiment Result")


                st.success(
                    f"Prediction: {result['label']}"
                )


                st.metric(
                    "Confidence",
                    f"{result['confidence']}%"
                )



            # NER

            elif task == "Named Entity Recognition":


                result = extract_entities(text)


                st.subheader("🔎 Entities")


                if result:

                    for entity in result:

                        st.write(
                            f"""
                            **{entity['word']}**
                            → {entity['entity']}
                            """
                        )

                else:

                    st.warning(
                        "No entities found"
                    )



            # Summarization

            elif task == "Text Summarization":


                result = summarize_text(text)


                st.subheader("📄 Summary")


                st.markdown(
                    f"""
                    <div class="result-box">

                    {result}

                    </div>
                    """,
                    unsafe_allow_html=True
                )



            # Translation

            elif task == "Translation":


                result = translate_text(text)


                st.subheader("🌐 Translation")


                st.success(result)



            # QA

            elif task == "Question Answering":


                if question:


                    result = answer_question(
                        text,
                        question
                    )


                    st.subheader("💡 Answer")


                    st.success(result)


                else:

                    st.warning(
                        "Enter a question first"
                    )



            # Generation

            elif task == "Text Generation":


                result = generate_text(text)


                st.subheader("✍️ Generated Text")


                st.write(result)



            # Zero Shot

            elif task == "Zero-Shot Classification":


                label_list = [
                    x.strip()
                    for x in labels.split(",")
                ]


                result = classify_text(
                    text,
                    label_list
                )


                st.subheader(
                    "🎯 Classification Result"
                )


                for label, score in zip(
                    result["labels"],
                    result["scores"]
                ):

                    st.progress(
                        float(score)
                    )

                    st.write(
                        f"""
                        **{label}**
                        
                        Confidence:
                        {score:.4f}
                        """
                    )



    else:


        st.warning(
            "Please enter some text"
        )