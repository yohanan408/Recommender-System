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

### Preventing Feature Leakage via Out-of-Fold (OOF) Stacking
Training a secondary booster directly on the predictions of a base model trained on the same data introduces massive target leakage, causing the tree model to overfit to the collaborative scores. 

To resolve this, this pipeline utilizes a **5-Fold Cross-Validation loop** using `scikit-surprise`. The model is trained iteratively on 4 folds to predict on the remaining 1 holdout fold. This creates a completely un-leaked `oof_svd_pred` vector for the entire 10M training subset, forcing `LightGBM` to learn how to correct SVD's natural, unseen prediction errors alongside the movie features.

---

## Feature Engineering & Optimization Highlights

* **Memory Space Management:** The interaction matrices are explicitly downcast from default 64-bit descriptors to 32-bit types (`int32`/`float32`), slashing the live RAM memory footprint by **50%** to safeguard execution stability on huge tables.
* **Macro Budget Correction:** Cleaned statistical profiles revealed a heavy data collection anomaly where missing budgets were pre-filled with a flat `12,000,000.0` value. The pipeline screens these artifacts out and dynamically imputes them using director-grouped structural means.
* **Composite Market Archetypes:** Multi-genre tags are kept intact as joined strings (e.g., `Action Adventure Animation Comedy Fantasy`) rather than exploded tokens. This preserves the distinct economic cost signatures attached to heavy VFX crossover blockbusters.

---

## 📁 Repository Structure

```filename
├── data/
│   ├── train.csv                # 10,000,038 explicit user rating interactions
│   ├── test.csv                 # 5,000,019 blind leaderboard coordinate rows
│   ├── imdb_data.csv            # Production budgets, runtimes, directors, and cast logs
│   ├── movies.csv               # Comprehensive movie catalog with multi-genre labels
│   ├── tags.csv                 # 1.09M crowd-sourced user-applied mood markers
│   ├── genome_scores.csv        # 15.5M tag relevance coefficient metrics
│   └── genome_tags.csv          # Tag numerical vocabulary lookups
├── pipeline.ipynb               # Production Jupyter notebook detailing EDA and model steps
├── submission.csv               # Final optimized prediction upload file
└── README.md                    # System documentation
```

---

## Step-by-Step Installation & Run Guide

### 1. Clone the Repository
```bash
git clone https://github.com
cd movie-recommendation-stacking
```

### 2. Install Required Environment Dependencies
```bash
pip install numpy pandas scikit-learn scikit-surprise lightgbm matplotlib seaborn kagglehub
```

### 3. Fetch Data & Execute Python Script
To trigger the automated server data synchronization loop and verify local data pathways, run the notebook or script containing your initialization code:
```python
import kagglehub

# Downloads competition files directly to local workspace environment path
path = kagglehub.competition_download('movie-recommendation-hackathon-2026')
print("Dataset successfully stored at:", path)
```

The script will clean the dataset fragments separately to conserve cache overhead, spin up the 5-fold array splitting workers, fit the underlying matrices, and save the optimized output data frame directly to `submission.csv`.
