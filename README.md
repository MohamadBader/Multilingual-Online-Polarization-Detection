# Multilingual-Online-Polarization-Detection
**SemEval-2026 Task 9 – Subtask 1**

This repository contains the full pipeline and codebase for **detecting multilingual, multicultural, and multi-event online polarization** using a **teacher–student knowledge distillation framework**.  
The project was developed as part of **CS445** and submitted to **POLAR @ SemEval-2026 Task 9**.

## 📌 Project Overview
Online polarization detection aims to identify short user-generated texts that express a strong **“us vs. them” hostile stance** toward social or political issues.  
The task is framed as **binary sentence-level classification (polarized vs. non-polarized)** across **22 languages** with diverse scripts and resource levels.

To address cross-lingual generalization and efficiency constraints, we adopt a **single-teacher distillation pipeline**:
- **Teacher**: XLM-RoBERTa Large (trained on high-resource languages)
- **Student**: XLM-RoBERTa Base (used alone at inference)

Only the student model is deployed at test time, ensuring **efficient inference** while retaining multilingual robustness learned from the teacher :contentReference[oaicite:0]{index=0}.

---

## 🧠 Methodology Summary

### 1. Task & Dataset
- **Task**: Binary polarization detection (Yes / No)
- **Languages**: 22
- **Training data**: `master_dataset.csv` (77,368 rows)
- **No external datasets used**

---

### 2. Deterministic Preprocessing
Each row is assigned a **deterministic ID**:
row_id = sha256(lang + "||" + text)[:16]
This enables:
- Safe sharding for parallel translation
- Merge-safe preprocessing
- Reproducible experiments :contentReference[oaicite:1]{index=1}.

---

### 3. Translation Pipeline (`text_en`)
- Non-teacher languages are translated to English
- Teacher languages copy `text → text_en`
- Translation executed in **parallel shards**
- 5,768 rows remained untranslated due to rate limits

Missing translations are handled safely during training via **partial distillation** (hard-label only) :contentReference[oaicite:2]{index=2}.

---

### 4. Synthetic Data Generation (Teacher Training)
- Synthetic data generated **only from polarized seeds**
- For each seed:
  - 2 polarized paraphrases
  - 2 neutral rewrites
- Final synthetic rows: **26,121**
- Leakage-safe grouping ensured via `source_row_id`

Final teacher training size: **103,489 rows** :contentReference[oaicite:3]{index=3}.

---

### 5. Teacher Model (XLM-R Large)
- Trained on **7 high-resource languages**:
  eng, spa, deu, rus, tur, pol, arb
- Group-aware, label-stratified validation split
- Trained for **3 epochs**

A second topic-based teacher was **removed** after detecting **label leakage** :contentReference[oaicite:4]{index=4}.

---

### 6. Knowledge Distillation
- Teacher produces **soft targets (logits)**
- Distillation applied **only when `text_en` exists**
- Synthetic samples excluded from distillation targets
- Teacher outputs merged into:
  master_with_teacherA_logits.csv

---

### 7. Student Model (XLM-R Base)
- Input: original multilingual text
- **Mixed supervision**:
- Hard-label loss (always)
- KL-divergence distillation loss (when teacher logits exist)
- Single model used for all 22 languages at inference

This **partial distillation strategy** preserves data coverage and prevents invalid supervision :contentReference[oaicite:5]{index=5}.

---

## 📊 Results

### Internal Validation (Student)
- **Macro F1**: 0.810
- **Accuracy**: 0.810
- **PR-AUC**: 0.886

Best model selected at **Epoch 3**.

---

### Official Codabench Development Evaluation (22 Languages)
- **Average Macro F1**: **0.783**
- **Average Accuracy**: **0.816**

Demonstrates strong multilingual generalization with efficient student-only inference :contentReference[oaicite:6]{index=6}.

---

## 🗂 Repository Structure
.

├── CS445_Data_Preprocessing&Generation_Pipeline.ipynb

├── CS445_Translation_Pipeline_P{0-4}_S{0-1}.ipynb

├── Teacher_Training_Pipeline.ipynb

├── Teacher_logits_generation_Pipeline.ipynb

├── Student_Training_Pipeline.ipynb

├── CS445_Submission.ipynb

├── master_with_teacherA_logits.csv

└── README.md

---

## 🚀 Inference & Submission
- Student model generates predictions per language:
  pred_<lang>.csv
- All files zipped and submitted to Codabench
- Teachers are **not used at inference time**

---

## 📚 References
Key references include XLM-R, knowledge distillation, and POLAR @ SemEval-2026.  
Full bibliography is available in the project report.

---

## 📝 License
This repository is released for **academic and research use only**.




