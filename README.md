# ArvyaX Emotion Intelligence System

This project builds a pipeline that reads user journal reflections after immersive sessions and predicts their emotional state, intensity, and recommends what they should do and when.

---

## Setup

Install dependencies:
```bash
pip install scikit-learn xgboost pandas numpy scipy joblib flask openpyxl
```

Python 3.8+ needed. Everything runs locally, no external APIs used.

---

## How to Run

### Train + predict in one step
```bash
python train.py --train data/Sample_arvyax_reflective_dataset.xlsx --test data/arvyax_test_inputs_120.xlsx
```

### Just predict (if model already trained)
```bash
python predict.py --model model_artifacts --test data/arvyax_test_inputs_120.xlsx --output predictions.csv
```

### Run the local API (bonus)
```bash
python app.py
```

---

## Project Structure
```
├── pipeline.py        # feature engineering, models, decision logic
├── train.py           # training script with eval + ablation
├── predict.py         # generates predictions.csv from test data
├── app.py             # simple local Flask API
├── data/              # put train and test files here
├── model_artifacts/   # saved model (auto-created)
├── predictions.csv    # output
├── ERROR_ANALYSIS.md
└── EDGE_PLAN.md
```

---

## Approach

### Why this isn't a standard classification problem
The data is messy — short texts, missing values, contradictory signals. The goal isn't just high accuracy, it's building something that reasons sensibly under uncertainty and gives useful output.

### Feature Engineering

**Text:** TF-IDF bigrams (300 features) + a small keyword lexicon to count positive, negative, and hedging words (but, still, idk, though). Also track text length since very short entries like "ok" or "fine" carry almost no signal.

**Metadata:** sleep, stress, energy, time of day, previous mood, face hint, reflection quality — all numerically encoded. Added interaction features: `stress_energy_ratio`, `sleep_stress` (sleep quality × inverse stress), and a `conflict_signal` flag for when face emotion contradicts reflection quality.

### Models

- **Emotion state** → Calibrated Random Forest. Calibration is important here because I need reliable probability outputs for the uncertainty score, not just hard predictions.
- **Intensity** → Gradient Boosting. Treated as ordinal classification (1–5 discrete classes). Regression would give meaningless fractional outputs.

I didn't go with deep learning because the dataset is ~1200 rows. Fine-tuning BERT or similar would overfit badly at this size.

### Decision Engine

Rule-based logic using predicted state, intensity, stress, energy, and time of day.

What to do mapping:
- overwhelmed + intensity ≥ 4 → box_breathing
- overwhelmed + intensity < 4 → grounding
- restless + high energy → movement
- restless + low energy → journaling
- calm + energy ≥ 3 → deep_work
- focused → deep_work
- mixed + high stress → journaling
- neutral → light_planning or rest depending on energy

Override: stress ≥ 5 AND energy ≤ 2 → rest regardless of state

When to do it is driven by time of day + urgency. High intensity overwhelmed/restless → now. Morning → within 15 min. Afternoon → later_today. Evening → tonight. Night → tomorrow_morning.

### Uncertainty

`confidence` = max class probability from the calibrated model.

`uncertain_flag = 1` if:
- confidence < 0.45
- text is under 5 words
- reflection_quality is "conflicted"

The model being explicit about when it's unsure is more useful than pretending to be confident on noisy inputs.

### Robustness

- Short text ("ok", "fine") → padded with neutral filler + flagged uncertain
- Missing numeric fields → imputed with training medians
- Missing categorical → mapped to "unknown" label
- Unseen categories at test time → mapped to -1 (RF handles this fine)

---

## Output

`predictions.csv` columns: id, predicted_state, predicted_intensity, confidence, uncertain_flag, what_to_do, when_to_do

`predictions_full.csv` adds a supportive_message column (bonus).
