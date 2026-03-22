# ERROR_ANALYSIS.md — ArvyaX Emotion Intelligence System

Analysis of failure cases, root causes, and improvement strategies.

---

## Overview

Errors cluster into four categories:
1. **Conflicting signal errors** — text says one thing, metadata says another
2. **Short-text ambiguity** — too little language to classify reliably
3. **Label noise** — subjective ground truth with borderline emotional states
4. **Temporal mismatch** — time-of-day creates context the model misreads

---

## Failure Case 1 — "Mixed" Predicted as "Calm"

**Input:** "The forest session made me calmer, but part of me still feels uneasy. Part of me wants rest, part of me wants action."
**Actual:** `mixed` | **Predicted:** `calm`
**Confidence:** 0.41 → uncertain_flag = 1 ✓

**Why it failed:**
The text opens with positive framing ("made me calmer") and the TF-IDF model anchors on "calmer" and "settled". The adversative "but" and "still feels uneasy" are underweighted by TF-IDF since they appear in many different contexts.

**How to improve:**
- Add discourse connective features: count "but", "though", "yet", "still" as explicit "hedging" signals
- Weight sentence position — the second half of a reflection often contradicts the first

---

## Failure Case 2 — Short Reflection Misclassified

**Input:** "ok"
**Actual:** `neutral` | **Predicted:** `restless` (random due to no text signal)
**Confidence:** 0.31 → uncertain_flag = 1 ✓

**Why it failed:**
Almost zero linguistic content. The model falls back on metadata, and with moderate stress (3) and energy (3), the slight random forest variation pushed toward "restless".

**How to improve:**
- For texts under 3 words, return a forced "neutral" prediction with `uncertain_flag = 1` and explicit low confidence
- Prompt the user for a richer reflection in a real product

---

## Failure Case 3 — Overwhelmed Predicted as Restless

**Input:** "I sat through the cafe ambience, but I still feel flooded by what I need to do. I'm carrying too much in my head."
**Actual:** `overwhelmed` | **Predicted:** `restless`
**Confidence:** 0.38 → uncertain_flag = 1 ✓

**Why it failed:**
"Flooded" and "carrying too much" are strong overwhelm markers but appear rarely in the dataset. "Still feel" and "too much" overlap with restless patterns. With only ~100 training samples, rare expressions don't generalize.

**How to improve:**
- Expand synonym mapping: "flooded" → overwhelm, "carrying too much" → overwhelm
- Use domain-specific emotion lexicon (e.g., NRC Emotion Lexicon, locally bundled)

---

## Failure Case 4 — Confident Wrong Prediction

**Input:** "I feel mentally clear after the mountain session and ready to tackle one thing at a time."
**Actual:** `focused` | **Predicted:** `calm` | **Confidence:** 0.61 → uncertain_flag = 0

**Why it failed:**
"Clear", "settled", "ready" are shared between `calm` and `focused`. Without metadata showing low stress + high energy, the model leans toward calm. In this case stress=2 and energy=3 — borderline. The model was confidently wrong.

**How to improve:**
- Confidence calibration: look at cases where confidence > 0.55 but model is wrong
- Add a "task-oriented" keyword feature: presence of "tackle", "plan", "hardest task", "begin" → focused
- Investigate post-hoc calibration curves

---

## Failure Case 5 — Contradictory Metadata (Calm Face + Overwhelmed Text)

**Input:** "I wanted the mountain to calm me, but today my stress feels bigger than the session. I feel emotionally tired."
**face_emotion_hint:** `calm_face` | **stress_level:** 5
**Actual:** `overwhelmed` | **Predicted:** `mixed`

**Why it failed:**
`calm_face` dragged the prediction toward `mixed` because the model partially trusts the face signal. But in real life, people mask distress facially — especially in semi-public settings (café context).

**How to improve:**
- Downweight `face_emotion_hint` when text explicitly describes high stress/overwhelm
- Add a `face_text_contradiction` feature: face score < 3 AND neg_kw > 3 → flag

---

## Failure Case 6 — Intensity Underestimated

**Input:** "Even after the forest track, I feel exhausted and emotionally overloaded."
**Actual intensity:** 5 | **Predicted intensity:** 3

**Why it failed:**
The text is short (1 sentence). The model saw similar phrasing in lower-intensity training samples. Without metadata backing (low sleep, high stress), the regression/classification hedged toward the center of the scale.

**How to improve:**
- Weight extreme intensity labels more heavily in training (class_weight for intensity classes 1 and 5)
- Add explicit marker: "emotionally overloaded" + "exhausted" in same sentence → intensity ≥ 4 override

---

## Failure Case 7 — Night-time Restlessness Missed

**Input:** "I couldn't really settle into the cafe track; I kept thinking of everything at once."
**time_of_day:** night | **sleep_hours:** 4.5 | **stress_level:** 5
**Actual:** `restless` intensity 5 | **Predicted:** `overwhelmed` intensity 4

**Why it failed:**
At night with 4.5h sleep and stress=5, metadata strongly signals overwhelm. But the journal is behaviorally describing restlessness ("couldn't settle", "kept thinking"). The model weighted metadata too heavily here.

**How to improve:**
- Text patterns like "couldn't settle", "kept thinking", "everything at once" → strong restless signals
- Consider ensemble that votes between text-dominant and metadata-dominant predictions

---

## Failure Case 8 — Label Noise: "Neutral" vs "Mixed"

**Input:** "The forest session was okay. I don't feel much different, just a bit more aware. Nothing really clicked yet."
**Actual:** `neutral` | **Predicted:** `mixed`

**Why it failed:**
Ground truth says neutral, but the text contains some hedging ("just a bit", "nothing really clicked") which the model reads as mixed-signal language. This is arguably a label noise case — a human annotator could legitimately label this either way.

**How to improve:**
- Use label smoothing during training for adjacent classes (neutral ↔ mixed, calm ↔ neutral)
- Consider merging neutral and mixed into a "low-activation" class for some use cases

---

## Failure Case 9 — Missing `previous_day_mood` Creates Drift

**Input:** *(previous_day_mood is blank)*
**Actual:** `overwhelmed` | **Predicted:** `mixed`

**Why it failed:**
When `previous_day_mood` is missing, the model fills it with "neutral". But if someone was overwhelmed yesterday too, that context would strongly signal ongoing overwhelm. Neutral imputation creates a false "reset".

**How to improve:**
- Don't just impute — add a `prev_mood_missing` binary flag as a feature
- Missing mood → slight upward pressure on uncertainty score
- In a real product, track user history to carry forward meaningful defaults

---

## Failure Case 10 — Decision Engine: Wrong "When" for Calm at Night

**Input:** state=`calm`, intensity=2, time_of_day=`night`
**Predicted:** `what_to_do: deep_work`, `when_to_do: tomorrow_morning`
**Better answer:** `what_to_do: rest`, `when_to_do: tonight`

**Why it failed:**
The decision engine correctly routes `calm + low intensity` to "later_today", then overrides to "tomorrow_morning" for night. But telling a calm person at night to do deep_work tomorrow morning is slightly off — rest or light journaling would better serve them.

**How to improve:**
- Add night-specific overrides: if `time_of_day == night AND state in (calm, neutral)` → `what: rest OR light journaling`, `when: tonight`
- Make the time-of-day rules more granular (separate night vs. late night)

---

## Summary Table

| # | State | Main Issue | Uncertain Flag Caught? | Fix Category |
|---|---|---|---|---|
| 1 | mixed→calm | Adversative text hedging | ✓ Yes | Text feature |
| 2 | neutral→restless | Short text | ✓ Yes | Robustness |
| 3 | overwhelmed→restless | Rare phrasing | ✓ Yes | Vocabulary |
| 4 | focused→calm | Confident wrong | ✗ No | Calibration |
| 5 | overwhelmed→mixed | Face contradiction | ✗ No | Feature design |
| 6 | intensity 5→3 | Short text + no metadata | ✗ No | Intensity weighting |
| 7 | restless→overwhelmed | Metadata overrides text | ✗ No | Ensemble balance |
| 8 | neutral→mixed | Label noise | ✗ No | Label smoothing |
| 9 | overwhelmed→mixed | Missing prev mood | ✓ Partial | Imputation |
| 10 | Decision error (calm/night) | Rule gap | N/A | Decision engine |

**Key insight:** The uncertainty flag catches ~50% of errors. The remaining 50% are overconfident mistakes — the hardest category to fix. These require better calibration, domain lexicons, and richer training data.
