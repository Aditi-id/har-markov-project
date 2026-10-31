# Human Activity Tracker — Markov Network (walking / running / sitting / standing / stairs)

## Run
```bash
pip install -r requirements.txt
python train.py            # downloads MotionSense (GitHub), trains; falls back to synthetic data if offline
streamlit run app.py
```
(`app.py` trains automatically on first launch if no model exists. `python train.py --synthetic` = pipeline test only.)

## Live tracking from your phone
Phone motion sensors need **HTTPS**. Easiest: run `streamlit run app.py`, then expose it with
`cloudflared tunnel --url http://localhost:8501` (or ngrok), open the https link on the phone, choose
"📱 Phone sensors (live)", tap **Start**. iPhone will ask for motion permission.
No phone? Use "▶️ Demo replay" or upload a CSV (`ax,ay,az` in g, `gx,gy,gz` in rad/s).

## Pipeline
sensors (50 Hz) → 2.56 s windows, 1 s hop → 11 orientation-invariant features → quantile discretisation (8 states)
→ **pairwise Markov network** (unary factors φ(x_i,Y) + Chow-Liu tree pairwise factors φ(x_i,x_j,Y), weight tuned on
validation people, temperature-calibrated) → HMM-style temporal filter → timeline + time per activity.

## Syllabus mapping
Unit I: naive-Bayes baseline (`pair_weight=0`) · Unit II: factors, local independencies (tree edges, Model tab),
joint distribution, classification, **Gibbs sampling** (`MarkovNet.gibbs`) · Unit III: add an RBM on the same features (extension).

## Notes
Split is by person (no leakage). Sitting vs standing is hardest since features ignore phone orientation.
Datasets: MotionSense (has running; phone-style accelerometer+gyro). UCI-HAR has no running class.
