"""Fashion MNIST classifier - Streamlit front end."""

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from utils.model import (
    CLASS_NAMES,
    compute_test_metrics,
    load_model,
    load_test_data,
    predict,
    preprocess_pil_image,
    preprocess_test_image,
)

st.set_page_config(
    page_title="Fashion MNIST classifier",
    page_icon=":material/checkroom:",
    layout="wide",
)

# Native navbar logo: shows in the app header and atop the sidebar.
st.logo(":material/checkroom:", size="large")

# Scoped gradient accent for the primary call-to-action button only.
# (Native theming can set solid colors but not gradients, so this one
# deliberate, key-scoped style rule covers the gradient look the rest
# of the app gets natively from .streamlit/config.toml.)
st.html(
    """
    <style>
    .st-key-sample_btn button {
        background: linear-gradient(135deg, #2563EB 0%, #7C3AED 100%);
        border: none;
    }
    .st-key-sample_btn button p {
        color: #FFFFFF;
    }
    .st-key-sample_btn button:hover {
        background: linear-gradient(135deg, #1D4ED8 0%, #6D28D9 100%);
    }
    .st-key-sample_btn button:hover p {
        color: #FFFFFF;
    }
    </style>
    """
)

model = load_model()


def render_prediction(display_image, model_input, caption, actual_label=None):
    with st.container(border=True):
        col_image, col_result = st.columns([1, 1.5], vertical_alignment="center")
        with col_image:
            st.image(display_image, caption=caption, width="stretch")

        probs = predict(model, model_input)
        top_idx = int(np.argmax(probs))
        predicted_label = CLASS_NAMES[top_idx]
        confidence = probs[top_idx]

        with col_result:
            metric_cols = st.columns(2)
            metric_cols[0].metric("Predicted class", predicted_label)
            metric_cols[1].metric("Confidence", f"{confidence * 100:.1f}%")
            if actual_label is not None:
                if predicted_label == actual_label:
                    st.success(f"Matches the actual label: {actual_label}", icon=":material/check_circle:")
                else:
                    st.warning(f"Actual label is {actual_label}", icon=":material/error:")

        st.write("**Class probabilities**")
        prob_df = pd.DataFrame({"Probability": probs}, index=CLASS_NAMES).sort_values("Probability")
        st.bar_chart(prob_df, horizontal=True, color="#2563EB")


with st.sidebar:
    st.subheader("Fashion MNIST classifier", icon=":material/checkroom:", divider="blue")
    st.caption("A dense neural network trained on 60,000 Zalando product images.")

    with st.container(border=True):
        st.write("**Categories**")
        with st.container(horizontal=True, gap="small"):
            for name in CLASS_NAMES:
                st.badge(name, color="blue")

    st.write("**Get started**")
    with st.container(horizontal=True, gap="small"):
        st.badge("Upload a photo", icon=":material/upload_file:", color="violet")
        st.badge("Try a sample", icon=":material/shuffle:", color="violet")

    st.space("large")
    with st.container(horizontal=True, horizontal_alignment="distribute"):
        st.caption("Model: fashion_mnist.keras")
        st.caption("v1.0")
    st.caption("Tip: switch light/dark mode from the ⋮ menu, top right.")

st.html(
    """
    <div style="
        background: linear-gradient(135deg, #2563EB 0%, #4F46E5 55%, #7C3AED 100%);
        border-radius: 20px;
        padding: 2rem 2.25rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(37, 99, 235, 0.25);
        display: flex;
        align-items: center;
        gap: 1.25rem;
    ">
        <svg width="44" height="44" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="flex-shrink:0;">
            <path d="M8 2L4 5V8H6V22H18V8H20V5L16 2L12 4L8 2Z" fill="#FFFFFF" fill-opacity="0.95"/>
        </svg>
        <div>
            <div style="color:#FFFFFF; font-size:1.7rem; font-weight:700; line-height:1.2;">
                Fashion MNIST classifier
            </div>
            <div style="color:#E0E7FF; font-size:0.95rem; margin-top:0.35rem;">
                Upload a clothing photo or try a test-set sample and watch a neural network
                predict its category in real time.
            </div>
        </div>
    </div>
    """
)

tab_upload, tab_sample, tab_about = st.tabs(
    [
        ":material/upload_file: Upload image",
        ":material/shuffle: Try a sample",
        ":material/analytics: About the model",
    ]
)

with tab_upload:
    st.write("Upload a clear, front-facing photo of a single clothing item.")
    invert = st.toggle(
        "Photo has a light background",
        value=True,
        help=(
            "Fashion MNIST images show a light garment on a black background. "
            "Leave this checked for typical photos (dark item on a light background); "
            "uncheck it if your image already looks like a Fashion MNIST sample."
        ),
    )
    uploaded = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
        label_visibility="collapsed",
    )

    if uploaded is not None:
        pil_image = Image.open(uploaded)
        model_input, display_28 = preprocess_pil_image(pil_image, invert=invert)
        render_prediction(pil_image, model_input, caption="Your upload")
        with st.expander("See what the model actually sees (28 x 28 grayscale)", icon=":material/visibility:"):
            st.image(display_28, width=140)
    else:
        st.info("Upload an image to get a prediction.", icon=":material/upload:")

with tab_sample:
    st.write("Pull a random image straight from the Fashion MNIST test set.")

    if st.button("Get a random sample", icon=":material/shuffle:", type="primary", key="sample_btn"):
        st.session_state["sample_idx"] = int(np.random.randint(0, 10000))
    if "sample_idx" not in st.session_state:
        st.session_state["sample_idx"] = 0

    test_images, test_labels = load_test_data()
    idx = st.session_state["sample_idx"]
    raw_image = test_images[idx]
    model_input = preprocess_test_image(raw_image)
    actual_label = CLASS_NAMES[int(test_labels[idx])]

    st.caption(f"Test sample #{idx}")
    render_prediction(raw_image, model_input, caption=f"Actual: {actual_label}", actual_label=actual_label)

with tab_about:
    st.subheader("Architecture", icon=":material/hub:", divider="violet")
    st.write(
        "A fully-connected (dense) network trained from scratch on the 28x28 "
        "grayscale Fashion MNIST images, flattened to 784-length vectors."
    )
    st.code(
        "Input(784)\n"
        "  -> Dense(32, relu)\n"
        "  -> Dense(64, relu)\n"
        "  -> Dense(128, relu)\n"
        "  -> Dense(10, softmax)\n\n"
        "optimizer: rmsprop\n"
        "loss: sparse_categorical_crossentropy",
        language=None,
    )

    st.subheader("Performance on the test set", icon=":material/analytics:", divider="blue")
    accuracy, report = compute_test_metrics()
    with st.container(border=True):
        st.metric("Overall test accuracy", f"{accuracy * 100:.2f}%")

    per_class = {
        name: {
            "Precision": report[name]["precision"],
            "Recall": report[name]["recall"],
            "F1-score": report[name]["f1-score"],
            "Support": int(report[name]["support"]),
        }
        for name in CLASS_NAMES
    }
    report_df = pd.DataFrame.from_dict(per_class, orient="index").reset_index(names="Category")
    st.dataframe(
        report_df,
        column_config={
            "Precision": st.column_config.ProgressColumn("Precision", min_value=0, max_value=1, format="%.2f"),
            "Recall": st.column_config.ProgressColumn("Recall", min_value=0, max_value=1, format="%.2f"),
            "F1-score": st.column_config.ProgressColumn("F1-score", min_value=0, max_value=1, format="%.2f"),
            "Support": st.column_config.NumberColumn("Support"),
        },
        hide_index=True,
        width="stretch",
    )
    st.caption("Computed live from the current model against the full 10,000-image test set.")
