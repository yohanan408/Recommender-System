)](https://opensource.org)

An end-to-end machine learning pipeline built for the **Movie Recommendation Hackathon 2026**. The architecture implements a two-stage hybrid stacking ensemble that pipes Collaborative Matrix Factorization (`SVD`) outputs and processed IMDb metadata into a Gradient Boosting framework (`LightGBM`). The pipeline is engineered to eliminate validation data leakage while scaling efficiently over **15.5+ million data rows**.

---

## Challenge Description
In today’s technology-driven world, recommender systems are socially and economically critical for ensuring that individuals can make appropriate choices surrounding the content they engage with on a daily basis. One application where this is especially true surrounds movie content recommendations; where intelligent algorithms can help viewers find great titles from tens of thousands of options.

## Performance Benchmark

| Model Configuration | Validation RMSE | Improvement vs. Baseline |
| :--- | :---: | :---: |
| **Global Mean Baseline** | `1.0599` | *Reference* |
| **Movie Mean Baseline** | `0.9604` | + 9.38% |
| **Standalone Collaborative SVD (20 Latent Factors)** | `0.83017` | + 21.67% |
| **Final Hybrid LightGBM Stacking (OOF Clean)** | **`0.82896`** | **+ 21.79%** |

---

## Core Architecture & Pipeline Logic

Standard collaborative models struggle to integrate rich content metadata, whereas standard tabular models fail on raw sparse user/item interaction IDs. This framework bridges the gap with an isolated two-stage execution architecture:

