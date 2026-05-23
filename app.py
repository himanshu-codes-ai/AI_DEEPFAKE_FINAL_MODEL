"""
app.py — AIIthentic: Military-Grade Neural Forensics
Enhanced with: Auth, SQLite DB, Groq XAI, Feedback System, Admin Dashboard
Team CodePagloos
"""

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import cv2
import numpy as np
import tempfile
import os
import json
import time
import pandas as pd
from PIL import Image
from facenet_pytorch import MTCNN
from database import (
    init_db, create_user, authenticate_user, get_user_by_id,
    save_analysis, update_analysis_xai, get_analysis_by_id,
    get_user_history, save_feedback, has_feedback,
    get_platform_stats, get_all_analyses_admin,
)
from xai_explainer import generate_xai_explanation

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG & INIT
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AIIthentic | Neural Forensics",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model.pth"          # update if your path differs
TOP_K_FRAMES = 20
IMG_SIZE = 224

# ─────────────────────────────────────────────────────────────────────────────
#  CYBERPUNK CSS
# ─────────────────────────────────────────────────────────────────────────────
CYBERPUNK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@300;400;600;700&display=swap');

:root {
    --neon-green : #00ff88;
    --neon-red   : #ff2244;
    --neon-blue  : #00cfff;
    --neon-yellow: #ffdd00;
    --bg-dark    : #050810;
    --bg-card    : #0d1117;
    --border     : #1e2d3d;
    --text-dim   : #7a8fa6;
}

html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
    background-color: var(--bg-dark) !important;
    color: #c9d8e8;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060c14 0%, #0a1628 100%) !important;
    border-right: 1px solid var(--border);
}

/* Buttons */
.stButton > button {
    background: transparent;
    border: 1px solid var(--neon-green);
    color: var(--neon-green);
    font-family: 'Share Tech Mono', monospace;
    letter-spacing: 2px;
    text-transform: uppercase;
    transition: all 0.2s;
    border-radius: 2px;
}
.stButton > button:hover {
    background: var(--neon-green);
    color: #000;
    box-shadow: 0 0 20px rgba(0,255,136,0.4);
}

/* Text inputs */
.stTextInput input, .stTextArea textarea {
    background: #0a1120 !important;
    border: 1px solid var(--border) !important;
    color: #c9d8e8 !important;
    font-family: 'Share Tech Mono', monospace;
    border-radius: 2px !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--neon-blue) !important;
    box-shadow: 0 0 8px rgba(0,207,255,0.3) !important;
}

/* Progress bar */
.stProgress > div > div { background: var(--neon-green); }

/* Metric cards */
[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    padding: 1rem;
    border-radius: 4px;
}

/* File uploader */
[data-testid="stFileUploader"] {
    border: 1px dashed var(--border) !important;
    background: var(--bg-card) !important;
    border-radius: 4px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card);
    border-bottom: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] { color: var(--text-dim); }
.stTabs [aria-selected="true"] { color: var(--neon-blue) !important; border-bottom-color: var(--neon-blue) !important; }

/* Selectbox */
.stSelectbox > div > div {
    background: #0a1120 !important;
    border: 1px solid var(--border) !important;
    border-radius: 2px !important;
    color: #c9d8e8 !important;
}

/* Dataframe */
[data-testid="stDataFrame"] { border: 1px solid var(--border); }

/* Alert boxes */
.stAlert { border-left: 3px solid var(--neon-blue) !important; background: #0a1628 !important; }
.stSuccess { border-left-color: var(--neon-green) !important; }
.stError   { border-left-color: var(--neon-red) !important; }
.stWarning { border-left-color: var(--neon-yellow) !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-dark); }
::-webkit-scrollbar-thumb { background: #1e3050; border-radius: 3px; }
</style>
"""
st.markdown(CYBERPUNK_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  SESSION STATE BOOTSTRAP
# ─────────────────────────────────────────────────────────────────────────────
_defaults = {
    "logged_in"        : False,
    "user_id"          : None,
    "username"         : None,
    "role"             : "user",
    "page"             : "auth",          # auth | analyze | history | admin
    "last_analysis_id" : None,
    "awaiting_feedback": False,
    "last_result"      : None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL CLASSES  (keep in sync with your training code)
# ─────────────────────────────────────────────────────────────────────────────
class DeepfakeLSTM(nn.Module):
    def __init__(self, input_size=1536, hidden_size=512, num_layers=2, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        pooled = out.mean(dim=1)
        return self.classifier(pooled)


@st.cache_resource(show_spinner=False)
def load_models():
    """Load EfficientNet-B3 feature extractor + Bi-LSTM classifier."""
    efficientnet = models.efficientnet_b3(weights=None)
    efficientnet.classifier = nn.Identity()
    efficientnet = efficientnet.to(DEVICE).eval()

    lstm_model = DeepfakeLSTM().to(DEVICE).eval()

    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            # Combined checkpoint
            efficientnet_sd = {k.replace("efficientnet.", ""): v
                               for k, v in checkpoint["model_state_dict"].items()
                               if k.startswith("efficientnet.")}
            lstm_sd = {k.replace("lstm.", ""): v
                       for k, v in checkpoint["model_state_dict"].items()
                       if k.startswith("lstm.")}
            if efficientnet_sd:
                efficientnet.load_state_dict(efficientnet_sd, strict=False)
            if lstm_sd:
                lstm_model.load_state_dict(lstm_sd, strict=False)
        else:
            lstm_model.load_state_dict(checkpoint, strict=False)
    else:
        st.warning("⚠️  Model weights not found. Running in demo mode.")

    mtcnn = MTCNN(image_size=IMG_SIZE, margin=20, device=DEVICE, keep_all=False)
    return efficientnet, lstm_model, mtcnn


# ─────────────────────────────────────────────────────────────────────────────
#  INFERENCE PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def compute_entropy(frame1, frame2):
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist /= hist.sum() + 1e-9
    entropy = -np.sum(hist * np.log2(hist + 1e-9))
    return float(entropy)


def extract_top_frames(video_path: str, k: int = TOP_K_FRAMES):
    cap = cv2.VideoCapture(video_path)
    frames, scores = [], []
    prev = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if prev is not None:
            scores.append(compute_entropy(prev, frame))
            frames.append(frame)
        prev = frame.copy()
    cap.release()

    if not frames:
        return []

    top_indices = np.argsort(scores)[-k:]
    return [frames[i] for i in sorted(top_indices)]


def run_inference(video_path: str, efficientnet, lstm_model, mtcnn):
    t0 = time.time()
    top_frames = extract_top_frames(video_path)
    if not top_frames:
        return None, None, None, None

    features, frame_scores = [], []
    with torch.no_grad():
        for frame in top_frames:
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            face_tensor = mtcnn(pil)
            if face_tensor is None:
                tensor = TRANSFORM(pil).unsqueeze(0).to(DEVICE)
            else:
                if face_tensor.dim() == 3:
                    face_tensor = face_tensor.unsqueeze(0)
                tensor = face_tensor.to(DEVICE)

            feat = efficientnet(tensor)
            features.append(feat.cpu())

            # Per-frame score for XAI
            seq = feat.unsqueeze(0)
            logit = lstm_model(seq)
            prob = torch.sigmoid(logit).item()
            frame_scores.append(prob)

    if not features:
        return None, None, None, None

    seq_tensor = torch.stack(features, dim=1).to(DEVICE)      # (1, T, 1536)
    with torch.no_grad():
        logit = lstm_model(seq_tensor)
    conf = torch.sigmoid(logit).item()
    result = "DEEPFAKE" if conf > 0.5 else "REAL"
    proc_time = round(time.time() - t0, 2)
    return conf, result, frame_scores, proc_time


# ─────────────────────────────────────────────────────────────────────────────
#  UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def logo():
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;padding:12px 0 4px 0;border-bottom:1px solid #1e2d3d;margin-bottom:24px;">
        <span style="font-size:2rem;">👁️</span>
        <div>
            <div style="font-family:'Share Tech Mono',monospace;font-size:1.3rem;color:#00cfff;letter-spacing:3px;">AIIthentic</div>
            <div style="font-size:0.75rem;color:#7a8fa6;letter-spacing:2px;">NEURAL FORENSICS SUITE</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def verdict_banner(result: str, confidence: float):
    is_fake = result == "DEEPFAKE"
    color   = "#ff2244" if is_fake else "#00ff88"
    icon    = "⚠️" if is_fake else "✅"
    label   = "DEEPFAKE DETECTED" if is_fake else "AUTHENTIC FOOTAGE"
    pct     = confidence * 100 if is_fake else (1 - confidence) * 100
    st.markdown(f"""
    <div style="border:2px solid {color};border-radius:4px;padding:24px;text-align:center;
                background:{'rgba(255,34,68,0.08)' if is_fake else 'rgba(0,255,136,0.06)'};
                box-shadow:0 0 30px {'rgba(255,34,68,0.25)' if is_fake else 'rgba(0,255,136,0.2)'};
                margin:16px 0;">
        <div style="font-size:2.5rem;margin-bottom:8px;">{icon}</div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:1.8rem;color:{color};
                    letter-spacing:4px;font-weight:bold;">{label}</div>
        <div style="color:#7a8fa6;margin-top:8px;font-size:0.95rem;letter-spacing:2px;">
            CONFIDENCE: <span style="color:{color};font-weight:bold;">{pct:.1f}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def sidebar_nav():
    with st.sidebar:
        logo()
        if st.session_state.logged_in:
            st.markdown(f"""
            <div style="background:#0a1628;border:1px solid #1e2d3d;border-radius:4px;
                        padding:12px;margin-bottom:20px;font-size:0.9rem;">
                <div style="color:#7a8fa6;font-size:0.75rem;letter-spacing:2px;margin-bottom:4px;">OPERATOR</div>
                <div style="color:#00cfff;font-family:'Share Tech Mono',monospace;">
                    {st.session_state.username}
                </div>
                {"<div style='color:#ffdd00;font-size:0.7rem;letter-spacing:2px;margin-top:4px;'>⬛ ADMIN ACCESS</div>" if st.session_state.role == 'admin' else ""}
            </div>
            """, unsafe_allow_html=True)

            nav_items = [
                ("🔬  Analyze Video",  "analyze"),
                ("📂  My History",     "history"),
            ]
            if st.session_state.role == "admin":
                nav_items.append(("🛡️  Admin Dashboard", "admin"))

            for label, page in nav_items:
                active = st.session_state.page == page
                style = f"color:#00cfff;font-weight:bold;" if active else ""
                if st.button(label, key=f"nav_{page}", use_container_width=True):
                    st.session_state.page = page
                    st.session_state.awaiting_feedback = False
                    st.rerun()

            st.markdown("---")
            if st.button("🚪  Logout", use_container_width=True):
                for k in _defaults:
                    st.session_state[k] = _defaults[k]
                st.rerun()

        st.markdown("""
        <div style="position:absolute;bottom:20px;left:0;right:0;text-align:center;
                    color:#2a3d52;font-size:0.7rem;letter-spacing:1px;">
            Made with ❤️ by Team CodePagloos
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE: AUTH (Login / Signup)
# ─────────────────────────────────────────────────────────────────────────────
def page_auth():
    col_l, col_c, col_r = st.columns([1, 1.2, 1])
    with col_c:
        st.markdown("""
        <div style="text-align:center;padding:40px 0 32px 0;">
            <div style="font-size:3.5rem;">👁️</div>
            <div style="font-family:'Share Tech Mono',monospace;font-size:2rem;color:#00cfff;
                        letter-spacing:4px;margin-top:8px;">AIIthentic</div>
            <div style="color:#7a8fa6;letter-spacing:3px;font-size:0.8rem;margin-top:4px;">
                MILITARY-GRADE NEURAL FORENSICS
            </div>
        </div>
        """, unsafe_allow_html=True)

        tab_login, tab_signup = st.tabs(["LOGIN", "CREATE ACCOUNT"])

        with tab_login:
            st.markdown("<br>", unsafe_allow_html=True)
            username = st.text_input("Username", key="login_user", placeholder="operator_id")
            password = st.text_input("Password", type="password", key="login_pass", placeholder="••••••••")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("ACCESS SYSTEM", use_container_width=True, key="btn_login"):
                if not username or not password:
                    st.error("Please fill in all fields.")
                else:
                    user = authenticate_user(username, password)
                    if user:
                        st.session_state.logged_in  = True
                        st.session_state.user_id    = user["id"]
                        st.session_state.username   = user["username"]
                        st.session_state.role       = user["role"]
                        st.session_state.page       = "analyze"
                        st.success(f"Welcome back, {user['username']}!")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error("Invalid credentials.")

        with tab_signup:
            st.markdown("<br>", unsafe_allow_html=True)
            new_user  = st.text_input("Username",        key="reg_user",  placeholder="choose a handle")
            new_email = st.text_input("Email",           key="reg_email", placeholder="you@domain.com")
            new_pass  = st.text_input("Password",        type="password", key="reg_pass1", placeholder="min. 6 chars")
            new_pass2 = st.text_input("Confirm Password",type="password", key="reg_pass2", placeholder="repeat")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("JOIN THE NETWORK", use_container_width=True, key="btn_signup"):
                if not all([new_user, new_email, new_pass, new_pass2]):
                    st.error("All fields required.")
                elif new_pass != new_pass2:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = create_user(new_user, new_email, new_pass)
                    if ok:
                        st.success(msg + " Please log in.")
                    else:
                        st.error(msg)


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE: ANALYZE
# ─────────────────────────────────────────────────────────────────────────────
def page_analyze():
    efficientnet, lstm_model, mtcnn = load_models()

    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace;font-size:1.4rem;color:#00cfff;
                letter-spacing:3px;margin-bottom:4px;">FORENSIC ANALYSIS TERMINAL</div>
    <div style="color:#7a8fa6;font-size:0.85rem;margin-bottom:24px;">
        Upload a video. The neural pipeline will scan for synthetic artifacts.
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop video file here",
        type=["mp4", "avi", "mov", "mkv"],
        help="Supported: MP4, AVI, MOV, MKV  |  Max 200 MB",
    )

    if uploaded and not st.session_state.awaiting_feedback:
        st.video(uploaded)
        st.markdown("---")

        if st.button("⚡  INITIATE FORENSIC SCAN", use_container_width=True):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            progress = st.progress(0, text="Initialising entropy sampler…")
            status   = st.empty()

            try:
                progress.progress(15, text="Extracting high-motion frames…")
                time.sleep(0.2)
                progress.progress(40, text="Running spatial encoder (EfficientNet-B3)…")
                conf, result, frame_scores, proc_time = run_inference(
                    tmp_path, efficientnet, lstm_model, mtcnn
                )
                progress.progress(75, text="Bi-LSTM temporal analysis…")
                time.sleep(0.2)
                progress.progress(90, text="Generating forensic report…")

                if conf is None:
                    st.error("❌ No faces detected in the video. Please upload a video with a visible face.")
                    progress.empty()
                    return

                # Save to DB immediately (XAI will be patched after)
                analysis_id = save_analysis(
                    user_id          = st.session_state.user_id,
                    video_name       = uploaded.name,
                    result           = result,
                    confidence_score = conf,
                    frame_scores     = frame_scores,
                    top_frames_count = len(frame_scores) if frame_scores else 0,
                    processing_time  = proc_time,
                )

                progress.progress(100, text="Complete.")
                time.sleep(0.3)
                progress.empty()

                st.session_state.last_analysis_id = analysis_id
                st.session_state.last_result = {
                    "conf": conf, "result": result,
                    "frame_scores": frame_scores, "proc_time": proc_time,
                    "video_name": uploaded.name,
                }
                st.session_state.awaiting_feedback = True
                st.rerun()

            except Exception as e:
                progress.empty()
                st.error(f"Analysis error: {e}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    # ── RESULT DISPLAY ──────────────────────────────────────────────
    if st.session_state.awaiting_feedback and st.session_state.last_result:
        r           = st.session_state.last_result
        conf        = r["conf"]
        result      = r["result"]
        frame_scores= r["frame_scores"] or []
        proc_time   = r["proc_time"]
        video_name  = r["video_name"]
        analysis_id = st.session_state.last_analysis_id

        verdict_banner(result, conf)

        col1, col2, col3 = st.columns(3)
        col1.metric("Confidence Score", f"{conf:.4f}")
        col2.metric("Frames Analyzed", len(frame_scores))
        col3.metric("Processing Time", f"{proc_time}s")

        # Frame-level chart
        if frame_scores:
            st.markdown("#### 📈 Frame-Level Fake Probability")
            chart_df = pd.DataFrame({
                "Frame": range(1, len(frame_scores) + 1),
                "Fake Probability": frame_scores,
            }).set_index("Frame")
            st.line_chart(chart_df, color="#ff2244" if result == "DEEPFAKE" else "#00ff88")

        # ── XAI EXPLANATION ──────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 🧠 Forensic Explanation  *(Powered by Groq · Llama 3 70B)*")
        analysis_row = get_analysis_by_id(analysis_id)

        if analysis_row and analysis_row.get("xai_explanation"):
            st.markdown(analysis_row["xai_explanation"])
        else:
            groq_key = ""
            try:
                groq_key = st.secrets.get("GROQ_API_KEY", "")
            except Exception:
                pass
            if not groq_key:
                groq_key = os.environ.get("GROQ_API_KEY", "")

            with st.spinner("Generating forensic report via Groq…"):
                explanation = generate_xai_explanation(
                    confidence_score = conf,
                    result           = result,
                    video_name       = video_name,
                    frame_scores     = frame_scores,
                    groq_api_key     = groq_key,
                )
            update_analysis_xai(analysis_id, explanation)
            st.markdown(explanation)

        # ── FEEDBACK FORM ─────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 💬 Did We Get It Right?")
        if has_feedback(analysis_id, st.session_state.user_id):
            st.success("✅ You've already submitted feedback for this analysis. Thank you!")
        else:
            with st.container():
                fb_col1, fb_col2 = st.columns(2)
                with fb_col1:
                    is_correct = st.radio(
                        "Was the verdict correct?",
                        ["Yes — it got it right", "No — this was wrong"],
                        horizontal=True,
                        key="fb_correct",
                    )
                with fb_col2:
                    rating = st.select_slider(
                        "Rate the explanation quality",
                        options=[1, 2, 3, 4, 5],
                        value=4,
                        key="fb_rating",
                    )
                comment = st.text_area(
                    "Additional comments (optional)",
                    placeholder="Was there anything unusual the model missed?",
                    key="fb_comment",
                    max_chars=500,
                )
                if st.button("📩  SUBMIT FEEDBACK", use_container_width=True):
                    save_feedback(
                        analysis_id = analysis_id,
                        user_id     = st.session_state.user_id,
                        rating      = rating,
                        is_correct  = is_correct.startswith("Yes"),
                        comment     = comment.strip() or None,
                    )
                    st.success("Feedback recorded. Thank you for improving the forensic suite!")

        st.markdown("---")
        if st.button("🔄  Analyze Another Video", use_container_width=True):
            st.session_state.awaiting_feedback = False
            st.session_state.last_result       = None
            st.session_state.last_analysis_id  = None
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE: HISTORY
# ─────────────────────────────────────────────────────────────────────────────
def page_history():
    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace;font-size:1.4rem;color:#00cfff;
                letter-spacing:3px;margin-bottom:4px;">ANALYSIS HISTORY</div>
    <div style="color:#7a8fa6;font-size:0.85rem;margin-bottom:24px;">
        All videos you have submitted for forensic analysis.
    </div>
    """, unsafe_allow_html=True)

    history = get_user_history(st.session_state.user_id)
    if not history:
        st.info("No analyses yet. Upload your first video to get started.")
        return

    # Summary metrics
    total    = len(history)
    fakes    = sum(1 for h in history if h["result"] == "DEEPFAKE")
    reals    = total - fakes
    avg_conf = sum(h["confidence_score"] for h in history) / total
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Analyzed",   total)
    c2.metric("Deepfakes Found",  fakes)
    c3.metric("Authentic Videos", reals)
    c4.metric("Avg Confidence",   f"{avg_conf:.3f}")

    st.markdown("---")

    # Table
    rows = []
    for h in history:
        verdict_icon = "⚠️ DEEPFAKE" if h["result"] == "DEEPFAKE" else "✅ REAL"
        rows.append({
            "Date"       : h["created_at"][:16] if h["created_at"] else "—",
            "Video"      : h["video_name"],
            "Verdict"    : verdict_icon,
            "Confidence" : f"{h['confidence_score']:.4f}",
            "Time (s)"   : h["processing_time"] or "—",
            "Rating"     : "⭐" * h["rating"] if h["rating"] else "—",
            "Correct?"   : ("✅" if h["is_correct"] else "❌") if h["is_correct"] is not None else "—",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Expandable: view XAI for past analyses
    st.markdown("#### View Forensic Explanation")
    selected_id = st.selectbox(
        "Select analysis ID to view explanation",
        options=[h["id"] for h in history],
        format_func=lambda x: f"#{x} — {next(h['video_name'] for h in history if h['id']==x)}",
    )
    if selected_id:
        row = get_analysis_by_id(selected_id)
        if row and row.get("xai_explanation"):
            with st.expander("📄 Forensic Report", expanded=True):
                verdict_banner(row["result"], row["confidence_score"])
                st.markdown(row["xai_explanation"])
        else:
            st.info("No explanation available for this analysis.")


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE: ADMIN
# ─────────────────────────────────────────────────────────────────────────────
def page_admin():
    if st.session_state.role != "admin":
        st.error("Access denied.")
        return

    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace;font-size:1.4rem;color:#ffdd00;
                letter-spacing:3px;margin-bottom:4px;">⬛ ADMIN DASHBOARD</div>
    <div style="color:#7a8fa6;font-size:0.85rem;margin-bottom:24px;">
        Platform-wide analytics and user activity monitor.
    </div>
    """, unsafe_allow_html=True)

    stats = get_platform_stats()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Users",    stats["total_users"])
    c2.metric("Total Analyses", stats["total_analyses"])
    c3.metric("Deepfakes",      stats["total_fakes"])
    c4.metric("Avg Rating",     f"{stats['avg_rating']:.2f} ⭐")
    c5.metric("Model Accuracy", f"{stats['user_accuracy']:.1f}%")

    st.markdown("---")

    tab_analyses, tab_feedback = st.tabs(["ALL ANALYSES", "FEEDBACK OVERVIEW"])

    with tab_analyses:
        all_data = get_all_analyses_admin()
        if all_data:
            rows = [{
                "ID"         : d["id"],
                "User"       : d["username"],
                "Video"      : d["video_name"],
                "Verdict"    : d["result"],
                "Confidence" : f"{d['confidence_score']:.4f}",
                "Date"       : d["created_at"][:16] if d["created_at"] else "—",
                "Rating"     : d["rating"] or "—",
                "Correct?"   : ("✅" if d["is_correct"] else "❌") if d["is_correct"] is not None else "—",
            } for d in all_data]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No analyses yet.")

    with tab_feedback:
        all_data = get_all_analyses_admin()
        feedbacks = [d for d in all_data if d.get("rating")]
        if feedbacks:
            # Rating distribution
            ratings = [d["rating"] for d in feedbacks if d["rating"]]
            dist_df = pd.DataFrame({"Rating": ratings})
            st.bar_chart(dist_df["Rating"].value_counts().sort_index())
        else:
            st.info("No feedback collected yet.")


# ─────────────────────────────────────────────────────────────────────────────
#  ROUTER
# ─────────────────────────────────────────────────────────────────────────────
def main():
    sidebar_nav()

    if not st.session_state.logged_in:
        page_auth()
        return

    page = st.session_state.page
    if page == "analyze":
        page_analyze()
    elif page == "history":
        page_history()
    elif page == "admin":
        page_admin()
    else:
        st.session_state.page = "analyze"
        st.rerun()


if __name__ == "__main__":
    main()
