import os, joblib, numpy as np, pandas as pd, altair as alt, streamlit as st
from har.features import CODES, NAMES, EMOJI, COLORS, FS, HOP, WIN, windows
from har.tracker import Tracker
from har import sensor

st.set_page_config(page_title="Human Activity Tracker", page_icon="🏃", layout="wide")
if not os.path.exists("models/har_model.joblib"):
    with st.spinner("First run: training the model ..."):
        import train; train.main([])

@st.cache_resource
def load(): return joblib.load("models/har_model.joblib")
bundle = load(); model, demo = bundle["model"], bundle["demo"]; S = st.session_state
if "tr" not in S: S.tr, S.ptr, S.consumed, S.sess = Tracker(model), 0, 0, None

DOM, RNG = [NAMES[c] for c in CODES], [COLORS[c] for c in CODES]
col = alt.Scale(domain=DOM, range=RNG)

st.title("🏃 Human Activity Tracker — Markov Network")
st.sidebar.header("Input")
mode = st.sidebar.radio("Source", ["📱 Phone sensors (live)", "▶️ Demo replay", "📄 Upload CSV"])
stay = st.sidebar.slider("Temporal smoothing (stay probability)", .5, .98, .85, .01)
S.tr.stay = stay
if st.sidebar.button("Reset tracking"): S.tr.reset(); S.ptr = 0; S.consumed = 0; S.sess = None; st.rerun()
if S.tr.log:
    df_ = pd.DataFrame([dict(start_s=e["t0"], activity=NAMES[e["smooth"]], confidence=e["conf"]) for e in S.tr.log])
    st.sidebar.download_button("⬇️ Download activity log", df_.to_csv(index=False).encode(), "activity_log.csv", "text/csv")

def render(truth=False):
    L = S.tr.log
    if not L: st.info(f"Collecting data… need {WIN / FS:.1f} s, then one prediction per second."); return
    e = L[-1]; a, b = st.columns([1, 2])
    a.markdown(f"<div style='text-align:center;padding:18px;border-radius:16px;background:{COLORS[e['smooth']]}22;"
               f"border:2px solid {COLORS[e['smooth']]}'><div style='font-size:80px'>{EMOJI[e['smooth']]}</div>"
               f"<div style='font-size:36px;font-weight:700'>{NAMES[e['smooth']].upper()}</div>"
               f"<div>confidence {e['conf']:.0%}</div></div>", unsafe_allow_html=True)
    if truth and e["truth"]: a.caption(f"Actual (demo label): {NAMES[e['truth']]}")
    pr = pd.DataFrame(dict(Activity=[NAMES[c] for c in model.classes], Probability=e["probs"]))
    b.altair_chart(alt.Chart(pr).mark_bar().encode(x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1])), y=alt.Y("Activity:N", sort=DOM),
                   color=alt.Color("Activity:N", scale=col, legend=None)).properties(height=220), width="stretch")
    df = pd.DataFrame([dict(t0=x["t0"], t1=x["t1"], Activity=NAMES[x["smooth"]]) for x in L])
    st.subheader("Activity timeline")
    st.altair_chart(alt.Chart(df).mark_bar(height=22).encode(x=alt.X("t0:Q", title="seconds"), x2="t1:Q", y=alt.Y("Activity:N", sort=DOM),
                    color=alt.Color("Activity:N", scale=col, legend=None)).properties(height=200), width="stretch")
    st.subheader("Time spent per activity")
    tt = df.assign(sec=df.t1 - df.t0).groupby("Activity").sec.sum().reindex(DOM).fillna(0)
    cols = st.columns(len(DOM))
    for c, n in zip(cols, DOM): c.metric(n, f"{tt[n]:.0f} s")

if mode.startswith("📱"):
    st.caption("Open this app on your phone over **HTTPS** (see README), tap Start, and put the phone in your pocket or hand.")
    d = sensor.sensor_stream()
    if d:
        if d["session"] != S.sess: S.tr.reset(); S.sess, S.consumed = d["session"], 0
        end = d["i0"] + len(d["samples"]); new = d["samples"][max(0, S.consumed - d["i0"]):]
        if new: S.tr.push(new)
        S.consumed = end; st.caption(f"Sensor rate: {d['rate']} Hz")
    render()
elif mode.startswith("▶️"):
    run = st.toggle("Replay running", True)
    @st.fragment(run_every=1 if run else None)
    def live():
        if run:
            i = S.ptr % (len(demo) - FS); row = demo[i:i + FS]; S.ptr += FS
            S.tr.push(row[:, :6], truth=CODES[int(row[len(row) // 2, 6])])
        render(truth=True)
    st.caption("Replays unseen recordings of every activity (shuffled) through the same pipeline as live data.")
    live()
else:
    up = st.file_uploader("CSV with columns ax,ay,az (g) and gx,gy,gz (rad/s)", type="csv")
    if up:
        df = pd.read_csv(up); cols_ = ["ax", "ay", "az", "gx", "gy", "gz"]
        if not all(c in df for c in cols_): st.error("Missing columns: " + ", ".join(cols_))
        else: S.tr.reset(); S.tr.push(df[cols_].to_numpy(float)); render()

tab1, tab2 = st.tabs(["🧠 Model", "ℹ️ About"])
with tab1:
    m = model.meta; st.write(f"Training data: **{m['source']}** · pairwise-factor weight {m['pair_weight']} · temperature {model.temp}")
    st.dataframe(pd.DataFrame(m["results"]).T.rename(columns=dict(acc="Accuracy", f1="Macro-F1")).style.format("{:.3f}"))
    names = ["acc_mag_mean", "acc_mag_std", "acc_mag_range", "vertical_std", "horizontal_mean", "vert_dom_freq", "vert_peak_ratio", "vert_zero_cross", "gyro_mag_mean", "gyro_mag_std", "jerk_mean"]
    st.write("Learned Markov-network edges between features (Chow-Liu tree, class-conditional):")
    st.code("\n".join(f"{names[i]}  —  {names[j]}" for i, j in m["edges"]))
    if st.button("🎲 Gibbs-sample the demo window"):
        from har.features import window_features
        dist = model.gibbs(window_features(demo[2000:2000 + WIN, :6]), n=500)
        st.bar_chart(pd.Series({NAMES[k]: v for k, v in dist.items()}))
with tab2:
    st.markdown("Features are **orientation-invariant** (vertical vs horizontal motion are computed relative to the gravity direction), so the phone can be in any pocket position. Sitting vs standing is the hardest pair without orientation cues.")
