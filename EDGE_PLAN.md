# EDGE_PLAN.md — On-Device & Mobile Deployment

## Objective

Run the ArvyaX Emotion Intelligence System on mobile / edge devices with:
- No internet connection required
- Low latency (< 200ms inference)
- Minimal battery/CPU impact

---

## Current System Size Audit

| Component | Estimated Size |
|---|---|
| TF-IDF vectorizer (300 features) | ~150 KB |
| Random Forest (200 trees) | ~3–5 MB |
| Gradient Boosting (150 trees) | ~1–2 MB |
| Scaler + encoders | < 50 KB |
| **Total** | **~5–8 MB** |

This is small enough to ship on-device as-is.

---

## Mobile Deployment Strategy

### Option A — Direct Python Model (Android/iOS)
**Tools:** [Chaquopy](https://chaquopy.com/) (Android) or PySide6 (desktop)

1. Export `model_artifacts/` with `joblib`
2. Bundle into APK/IPA alongside app code
3. Load model once at app startup
4. Run `pipeline.predict()` on device

**Pros:** No rewrite needed, exact same code  
**Cons:** Python runtime adds ~30MB app size

---

### Option B — ONNX Export (Recommended for Production)
Convert sklearn models to ONNX format for cross-platform inference:

```python
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Export emotion model
initial_type = [("float_input", FloatTensorType([None, n_features]))]
onnx_model = convert_sklearn(rf_base, initial_types=initial_type)

with open("emotion_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

Then use `onnxruntime` on-device (available for Android/iOS/Raspberry Pi).

**ONNX runtime size:** ~1.5MB  
**Inference latency:** ~5–15ms on mid-range device

---

### Option C — ONNX + Quantization (Ultra-lightweight)
For very low-end devices:

```python
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic("emotion_model.onnx", "emotion_model_quantized.onnx",
                 weight_type=QuantType.QUInt8)
```

**Result:** ~60–70% size reduction, ~1–3% accuracy drop — acceptable tradeoff.

---

## On-Device Architecture

```
User writes journal reflection
         │
         ▼
[Text Preprocessing]  ← runs locally, ~1ms
         │
         ▼
[TF-IDF Vectorization]  ← 300 features, ~5ms
         │
         ▼
[Metadata Feature Engineering]  ← ~1ms
         │
         ▼
[ONNX Emotion + Intensity Models]  ← ~10ms
         │
         ▼
[Decision Engine]  ← pure Python rules, ~0ms
         │
         ▼
[Output: state, intensity, what, when, message]
         │
         ▼
[Uncertainty Flag → show confidence indicator in UI]
```

**Total latency estimate: < 50ms** on a mid-range smartphone (2021+)

---

## Latency Breakdown

| Step | Estimated Latency |
|---|---|
| Text cleaning | ~0.5ms |
| TF-IDF transform | ~3ms |
| Metadata engineering | ~1ms |
| RF inference (ONNX) | ~8ms |
| GB inference (ONNX) | ~5ms |
| Decision engine | ~0.1ms |
| **Total** | **~18ms** |

---

## Battery & CPU Impact

- Random Forest + TF-IDF is **CPU-only** — no GPU needed
- Inference only runs on demand (user submits reflection)
- No background polling or continuous processing
- Estimated battery impact: **< 0.01% per inference** on modern device

---

## Offline Considerations

### What works fully offline
- All inference (model is bundled in app)
- Decision engine
- Supportive message generation (template-based, no LLM)

### What needs internet (if implemented)
- Model updates (push new model version)
- Sending anonymized logs for retraining

### Offline model update strategy
- Use **differential model updates**: only retrain changed ensemble trees
- Or: ship a lightweight adapter (fine-tuned weights delta) for download

---

## Model Size Tradeoffs

| Approach | Size | Accuracy | Latency | Recommendation |
|---|---|---|---|---|
| Full sklearn (joblib) | ~6MB | Baseline | ~30ms | Dev/testing only |
| ONNX (float32) | ~2MB | Same | ~15ms | Production mobile |
| ONNX (quantized int8) | ~0.7MB | -1% | ~8ms | Low-end devices |
| Pruned RF (50 trees) | ~1MB | -2% | ~5ms | Wearables/IoT |

---

## Future: Tiny Local Language Model

For the supportive message generation (currently template-based), we could replace with a quantized SLM:

- **DistilGPT2** (82MB) — too large for mobile
- **TinyLlama-1.1B (GGUF Q4_K_M)** (~700MB) — viable for high-end phones
- **Phi-2 (2.7B Q4)** (~1.5GB) — tablet/desktop only

**Recommendation for V1:** Keep template-based messages. Ship SLM only for premium tier where device RAM ≥ 4GB.

---

## Summary

The current system is already **edge-ready** at ~6MB. The primary deployment path is:
1. Export to ONNX
2. Bundle with mobile app
3. Run inference fully on-device
4. Update model via background download (wifi only)

No user data needs to leave the device for inference — a meaningful privacy advantage for a mental health product.
