# A CNN-based Local-Global Self-Attention via Averaged Window Embeddings for Hierarchical ECG Analysis

This repository contains the official implementation of the paper:

**A CNN-based Local-Global Self-Attention via Averaged Window Embeddings for Hierarchical ECG Analysis**  
Preprint submitted to **ECML PKDD 2025 (Research Track)**.  
[📄 Read the paper](https://ecmlpkdd-storage.s3.eu-central-1.amazonaws.com/preprints/2025/research/preprint_ecml_pkdd_2025_research_1436.pdf)  

**Key idea.** Queries (**Q**) are built from **overlapping convolutional window embeddings** that are **averaged** to summarize local morphology; keys and values (**K, V**) are computed **globally** over the entire sequence. Each attention block **halves** the sequence length (hierarchical reduction), while a pooled + 1×1-conv residual path preserves stability. Convolutional projections implicitly encode position, so **no explicit positional encoding** is required.

---

## ✨ Main Contributions

- **Local-Global Attention (LGA):**  
  Locally informed queries (averaged from overlapping conv windows) attend to globally derived keys/values, capturing **fine-grained wave morphology** and **long-range rhythm dependencies** simultaneously.

- **Hierarchical architecture tailored for ECGs:**  
  A multi-scale **deep convolutional front-end** followed by stacked **LGA transformer blocks** that progressively downsample the sequence, aligning early layers with **waveform details** and deeper layers with **beat-to-beat** and **rhythm** structure.

- **No positional encodings needed:**  
  Convolutional projections provide sufficient positional bias; explicit APE/RPE did **not** improve overall macro F1.

---

## 🧪 Datasets & Protocol

- **Training/validation/dev:** **CODE-15** (public 15% subset of the CODE study), split **by patient**: 90% train, 5% validation (early stopping), 5% development for ablations/hyper-params.  
- **Testing:** **CODE-TEST** (827 ECGs), labeled by **consensus of 2–3 cardiologists** across 6 abnormalities: 1st-degree AV block, RBBB, LBBB, sinus bradycardia, atrial fibrillation, sinus tachycardia.

---

## 📈 Results (Macro Metrics on CODE-TEST)

| Metric     | LGA-ECG |
|------------|---------|
| Accuracy   | **0.994** |
| Precision  | **0.907** |
| Recall     | **0.872** |
| F1-score   | **0.885** |

- **State-of-the-art macro F1 (0.885)**, surpassing ResNet baselines, ECG-Transformer, BAT, ECG-DETR, and HiT.  
- **Per-class F1:** LGA-ECG leads in **ST, LBBB, AF, 1st-AVB**; **RBBB** is on par with the best baseline; **SB** remains the hardest class.  
- **Human comparison:** The model **outperforms medical students, emergency residents, and cardiology residents** on precision, recall, and F1.
