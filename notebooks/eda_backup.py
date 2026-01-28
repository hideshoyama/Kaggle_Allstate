import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# 1. Load Data
# 1. データの読み込み
# ---------------------------------------------------------
# Load train and test datasets
# トレーニングデータとテストデータを読み込みます
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

print(f'Train shape (学習データ): {train.shape}')
print(f'Test shape (テストデータ): {test.shape}')

# ---------------------------------------------------------
# 5. Correlation Analysis
# 5. 相関行列の分析
# ---------------------------------------------------------
# Check correlation between continuous variables and the target 'loss'.
# 連続値の変数 (cont1 ~ cont14) と、目的変数 (loss) の関係を確認します。
# High correlation suggests similar features.
# 相関係数が高い（色が濃い）変数は、似た性質を持っている可能性があります。

# List features containing 'cont' (continuous features)
# 列名に 'cont' を含むもの（連続値）をリストアップ
cont_features = [col for col in train.columns if 'cont' in col]

# Calculate correlation between continuous features and 'loss'
# 連続値 + loss の相関係数を計算
corr = train[cont_features + ['loss']].corr()

# Visualize with a heatmap
# ヒートマップで可視化
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix (Continuous Features + Loss)\n相関行列（連続値変数 ＋ 損害額）')
plt.show()

# ---------------------------------------------------------
# 6. Scatter Plots: Continuous Features vs Loss
# 6. 散布図の確認: 連続値変数 vs 損害額
# ---------------------------------------------------------
# Plot each continuous feature (x) against loss (y).
# 各連続値変数 (X軸) と 損害額 loss (Y軸) の関係をプロットします。
# Sampling 1000 points to avoid overplotting.
# データ点が多すぎると重くて見づらいため、ランダムに1000件だけ抽出(sample)して表示します。

fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(15, 30))

for i, col in enumerate(cont_features):
    row = i // 2     # Row index / 行番号 (0~6)
    col_idx = i % 2  # Column index / 列番号 (0 or 1)
    
    # Plot scatter (alpha=0.5 for transparency)
    # 散布図の描画 (alpha=0.5 で少し透明にして重なりを見やすくする)
    sns.scatterplot(data=train.sample(1000), x=col, y='loss', ax=axes[row, col_idx], alpha=0.5)
    
    axes[row, col_idx].set_title(f'{col} vs Loss')

plt.tight_layout()
plt.show()
