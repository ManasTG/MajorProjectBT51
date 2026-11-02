# AI-Powered Patient Health Monitoring and Early Disease Prediction (PT051)

Phase-II starter for the arrhythmia-detection project described in the Phase-I report.

ECG -> band-pass filter -> R-peak detection -> beat features -> Random Forest (AAMI classes N/S/V/F/Q)
-> heart rate + abnormal-beat burden -> **Low / Moderate / High** risk and alerts -> dashboard + history.

## Run
```bash
pip install -r requirements.txt
python -m ml.train --source synthetic   # quick check, no download
python -m ml.train --source mitdb       # real data: downloads MIT-BIH, inter-patient DS1/DS2 split
uvicorn app.main:app --reload           # open http://127.0.0.1:8000
```
Upload a one-lead ECG as CSV (one sample per line, set the sampling rate), or use the demo buttons.

## Layout
- `ml/pipeline.py` preprocessing, R-peak detection, features, risk rules, `analyze()`
- `ml/train.py` training and evaluation (precision, recall, F1, confusion matrix, saved to `models/metrics.json`)
- `ml/synth.py` synthetic ECG for demos and tests
- `app/main.py` FastAPI API + SQLite (patients, analysis history); `app/static/index.html` dashboard

## Next steps
1. Train on MIT-BIH and report per-class recall (S and V matter most).
2. Compare against a 1D-CNN / LSTM (objective 2 in the report).
3. Replace the rule-based risk score with a calibrated one; add alert delivery (email/Telegram).
4. Wearable / live streaming input; extend to diabetes and other diseases.

Decision-support only, not a medical diagnosis.
