"""Task 7.1: Streamlit front end. Run from the repository root:  streamlit run app/app.py"""
import io, os, sys
import numpy as np
import pandas as pd
import streamlit as st
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from load_data import load_raw, CHANNELS
from model import HARNet

st.set_page_config(page_title="Activity recognition", layout="wide")

@st.cache_resource
def load_everything():
    net = HARNet().eval()
    net.load_state_dict(torch.load(f"{ROOT}/results/best_model.pt"))
    raw = load_raw(f"{ROOT}/data")
    return net, np.load(f"{ROOT}/results/norm_stats.npz"), raw["test"]["X"], raw["test"]["y"] - 1, raw["activity_names"]

net, stats, test_X, test_y, names = load_everything()
st.title("Which activity is this phone recording?")
window = None
with st.sidebar:
    source = st.radio("Input", ["Test window", "CSV upload"])
    if source == "Test window":
        kind = st.selectbox("Activity", range(6), format_func=lambda k: names[k])
        ids = np.flatnonzero(test_y == kind)
        window = test_X[ids[st.number_input(f"Window number (0 to {len(ids) - 1})", 0, len(ids) - 1, 0)]]
    else:
        file = st.file_uploader("128 rows x 9 columns, raw units, no header: " + ", ".join(CHANNELS), type=["csv", "txt"])
        if file is not None:
            try:
                window = np.loadtxt(io.StringIO(file.getvalue().decode()), delimiter=",")
                assert window.shape == (128, 9)
            except (ValueError, AssertionError):
                st.error("The CSV must contain 128 rows and 9 numeric columns, without a header.")
                window = None
if window is None:
    st.stop()                                                   # waiting for an upload

with torch.no_grad():
    logits, alpha = net(torch.tensor((window - stats["mean"]) / stats["std"], dtype=torch.float32)[None])
probs, alpha, seconds = logits.softmax(1)[0].numpy(), alpha[0].numpy(), np.arange(128) / 50
st.subheader(f"Predicted activity: {names[probs.argmax()]} ({probs.max():.1%})")
signals, outputs = st.columns(2)
with signals:
    for title, c in (("Body acceleration (g)", 0), ("Gyroscope (rad/s)", 3), ("Total acceleration (g)", 6)):
        st.caption(title + " over time (s)")
        st.line_chart(pd.DataFrame(window[:, c:c + 3], index=seconds, columns=CHANNELS[c:c + 3]), height=200)
with outputs:
    st.caption("Probability of each activity")
    st.bar_chart(pd.Series(probs, index=names), height=250)
    st.caption("Attention weights over time (s); flat attention would be 1/128 = 0.0078")
    st.line_chart(pd.Series(alpha, index=seconds), height=250)
