import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Install and import Japanese font support (Note: !pip commands usually only work in notebooks)
# 日本語フォント対応ライブラリのインポート
try:
    import japanize_matplotlib
except ImportError:
    print("Warning: japanize_matplotlib is not installed. Japanese characters may not display correctly.")

# ---------------------------------------------------------
# 1. Load Data
# 1. データの読み込み
# ---------------------------------------------------------
if os.path.exists('/kaggle/input/allstate-claims-severity/train.csv'):
    base_path = '/kaggle/input/allstate-claims-severity/'
    print('Running on Kaggle (Kaggle環境で実行中)')
else:
    base_path = '../input/'
    print('Running Locally (ローカル環境で実行中)')

train = pd.read_csv(base_path + 'train.csv')
test = pd.read_csv(base_path + 'test.csv')

print(f'Train shape (学習データ): {train.shape}')
print(f'Test shape (テストデータ): {test.shape}')

# ---------------------------------------------------------
# 5. Correlation Analysis
# 5. 相関行列の分析
# ---------------------------------------------------------
cont_features = [col for col in train.columns if 'cont' in col]
corr = train[cont_features + ['loss']].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix (Continuous Features + Loss)\n相関行列（連続値変数 ＋ 損害額）')
plt.show()

# ---------------------------------------------------------
# 6. Scatter Plots: Continuous Features vs Loss
# 6. 散布図の確認: 連続値変数 vs 損害額
# ---------------------------------------------------------
fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(15, 30))

for i, col in enumerate(cont_features):
    row = i // 2
    col_idx = i % 2
    sns.scatterplot(data=train.sample(1000), x=col, y='loss', ax=axes[row, col_idx], alpha=0.5)
    axes[row, col_idx].set_title(f'{col} vs Loss')

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 7. Categorical Features Analysis
# 7. カテゴリ変数の分析
# ---------------------------------------------------------
cat_features = [col for col in train.columns if 'cat' in col]
print(f'Number of categorical features (カテゴリ変数の数): {len(cat_features)}')

unique_counts = train[cat_features].nunique().sort_values(ascending=False)

plt.figure(figsize=(15, 6))
sns.barplot(x=unique_counts.index[:30], y=unique_counts.values[:30])
plt.xticks(rotation=90)
plt.title('Number of Unique Values per Categorical Feature (Top 30)\nカテゴリ変数ごとのユニークな値の数（上位30個）')
plt.show()

print(unique_counts.head(5))

# ---------------------------------------------------------
# 8. Analyzing cat116 vs Loss
# 8. cat116と損害額の関係
# ---------------------------------------------------------
# Get top 10 categories in cat116
# cat116の中で頻度が高い上位10個を取得
top10_categories = train['cat116'].value_counts().head(10).index

plt.figure(figsize=(15, 8))

# Draw Box Plot
# 箱ひげ図の描画
sns.boxplot(x='cat116', y='loss', data=train[train['cat116'].isin(top10_categories)], order=top10_categories)

plt.title('Loss Distribution by cat116 (Top 10 Frequent Categories)\ncat116（頻出上位10カテゴリ）ごとの損害額分布')
plt.yscale('log') # Log scale for better visibility / 対数軸で見やすくする
plt.show()
