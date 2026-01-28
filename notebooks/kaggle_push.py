from kaggle.api.kaggle_api_extended import KaggleApi
import os
import sys

# Windowsコンソールでの文字化け防止
sys.stdout.reconfigure(encoding='utf-8')

def push_kernel():
    api = KaggleApi()
    api.authenticate()
    
    kernel_dir = r"D:\Kaggle_Allstate\notebooks"
    print(f"ディレクトリからカーネルをアップロードしています: {kernel_dir}")
    
    try:
        api.kernels_push(kernel_dir)
        print("カーネルのアップロードに成功しました！")
        print("ステータス確認URL: https://www.kaggle.com/code/hideshoyama/Allstate-Claims-Severity-Analysis")
    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    push_kernel()
