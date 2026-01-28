# Allstate Claims Severity - Advanced Ensemble Regressor
**Kaggle Competition Project**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green.svg)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-orange.svg)
![Keras](https://img.shields.io/badge/DL-Keras%2FTensorFlow-red.svg)
![Optimization](https://img.shields.io/badge/Technique-Nelder--Mead-purple.svg)

## 📌 Project Overview / プロジェクト概要
Developed a high-performance regression model to predict the "loss" (claim severity) for Allstate Insurance.
Combining Gradient Boosting Trees (XGBoost, LightGBM) and Deep Learning (Neural Networks), achieved a robust ensemble model using automated weight optimization (Nelder-Mead/SLSQP).

Allstate（保険会社）の損害額（loss）を予測する回帰モデルを構築。
決定木モデル（XGBoost, LightGBM）とディープラーニング（Neural Network）を組み合わせ、さらに数学的最適化（scipy.optimize）を用いて「最適なアンサンブル重み」を自動算出するパイプラインを実装しました。

## 🏆 Key Achievements / 成果
- **Private Score (MAE):** 1129.77 (Approx. Top 35% / 3000 teams)
- **Improvement:** Reduced error by ~17 points from baseline (1146 -> 1129).
- **Technique:** Implemented a robust "3-Model Stacked Ensemble" that outperformed single models and simple averaging.

## 🛠️ Architecture / アーキテクチャ

```mermaid
graph LR
    Data["Raw Data"] --> Pre["Preprocessing<br>(Log Transform / One-Hot)"]
    
    Pre --> XGB["XGBoost<br>(Tree Logic)"]
    Pre --> LGB["LightGBM<br>(Speed & Accuracy)"]
    Pre --> NN["Neural Network<br>(Non-linear / Scaled)"]
    
    XGB --> Pred1["Prediction 1"]
    LGB --> Pred2["Prediction 2"]
    NN  --> Pred3["Prediction 3"]
    
    Pred1 --> Opt["Optimizer<br>(SLSQP / Nelder-Mead)"]
    Pred2 --> Opt
    Pred3 --> Opt
    
    Opt --> Final["Weighted Average<br>(Ensemble)"]
    Final --> Sub["Submission.csv<br>(MAE: 1129.77)"]
    
    style Opt fill:#f9f,stroke:#333,stroke-width:2px
    style Final fill:#bfb,stroke:#333,stroke-width:2px
```

## 💻 Technical Highlights / 技術的ハイライト

### 1. Hybrid Modeling (ハイブリッド・モデリング)
- **Diversity:** Combined "Tree-based" models (strong at categorical splits) with "Neural Networks" (strong at continuous scaling) to capture different data patterns.
- 「決定木が得意な論理的推論」と「ニューラルネットが得意な数学的推論」を組み合わせ、多様性を確保しました。

### 2. Automated Weight Optimization (重み最適化)
- Instead of manual guessing, used `scipy.optimize.minimize` to mathematically find the "Golden Ratio" of model weights based on validation data (OOF).
- 検証データ（Validation）に対する誤差が最小になるような重み配分（例: XGB 25%, NN 46%...）を自動計算し、人的バイアスを排除しました。

### 3. GPU Acceleration (GPU高速化)
- Enabled `device='cuda'` for XGBoost and `device='gpu'` for LightGBM/Keras to accelerate training on simple hardware.
- Kaggle環境のGPUをフル活用し、高速な実験サイクルを実現しました。

## 📂 Code Structure
- `analysis.ipynb`: Main experimentation notebook (EDA, Training, Optimization).
- `eda_backup.py`: Production-ready backup script.
