import pandas as pd
import numpy as np
import argparse
import os
from sklearn.cluster import KMeans

# 追加: 可視化用
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D描画で必要

def main():
    """
    【実行例】
    python3 cluster_csv.py input_data.csv output_data.csv --plot

    処理概要:
    1. CSVを読み込み(ID, pr_k, score, test_scoreなどを含む)
    2. 指定列を min-max スケーリング
    3. KMeans(k=3)でクラスタリング
    4. cluster列を追加して既存CSVを上書き or 新規ファイルに保存
    5. --plot が指定された場合は3次元散布図を表示 (pr_k_scaled, score_scaled, test_score_scaled)
    """
    parser = argparse.ArgumentParser(description="Clustering CSV data and append the result as 'cluster' column, with optional 3D plotting.")
    parser.add_argument("input_csv", help="Path to the input CSV file (must contain 'ID', 'pr_k', 'score', 'test_score' columns)")
    parser.add_argument("output_csv", nargs="?", default=None, help="Path to the output CSV file. If not specified, overwrite input_csv.")
    parser.add_argument("--plot", action="store_true", help="If specified, show a 3D scatter plot of the clustering result.")
    args = parser.parse_args()

    input_csv = args.input_csv
    output_csv = args.output_csv
    do_plot = args.plot

    if output_csv is None:
        # 出力先の指定がなければ、入力CSVを上書きする
        output_csv = input_csv

    if not os.path.exists(input_csv):
        print(f"Error: input_csv='{input_csv}' does not exist.")
        return

    # 1. CSVの読み込み
    df = pd.read_csv(input_csv)

    # ID列が存在しない場合はエラー
    if 'ID' not in df.columns:
        print("Error: 'ID' column not found in the input CSV.")
        return

    # クラスタリングに用いる列があるかチェック
    required_cols = ['pr_k', 'score', 'test_score']
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: '{col}' column not found in the input CSV.")
            return

    # 2. min-maxスケーリング
    #   pr_k: [1, 5] => (x - 1)/4
    #   score: [1, 5] => (x - 1)/4
    #   test_score: [0, 5] => x/5
    df['pr_k_scaled'] = (df['pr_k'] - 1.0) / (5.0 - 1.0)
    df['score_scaled'] = (df['score'] - 1.0) / (5.0 - 1.0)
    df['test_score_scaled'] = (df['test_score'] - 0.0) / (5.0 - 0.0)

    # 3. クラスタリング (k=3)
    kmeans = KMeans(n_clusters=3, random_state=42)
    X = df[['pr_k_scaled', 'score_scaled', 'test_score_scaled']].values
    kmeans.fit(X)
    cluster_labels = kmeans.labels_  # 0,1,2

    # 4. cluster列を追加してCSV出力
    df['cluster'] = cluster_labels

    # スケーリング列を消す場合
    df.drop(columns=['pr_k_scaled','score_scaled','test_score_scaled'], inplace=True)

    df.to_csv(output_csv, index=False)
    print(f"クラスタリング結果を '{output_csv}' に出力しました。")

    # 5. --plot が指定された場合は、クラスタリング結果を 3D プロット
    if do_plot:
        # 再度スケーリングデータを生成 (先ほど drop したので再計算)
        pr_k_scaled = (df['pr_k'] - 1.0) / 4.0
        score_scaled = (df['score'] - 1.0) / 4.0
        test_score_scaled = df['test_score'] / 5.0

        # プロット用の行列 X2
        X2 = np.column_stack((pr_k_scaled, score_scaled, test_score_scaled))
        labels_for_plot = df['cluster'].values  # 0,1,2

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("3D Clustering Result (k=3)")

        # クラスタごとに色分けして描画
        # (0,1,2) のラベルに対して適当に色割り当て
        colors = ['blue', 'green', 'red']
        for c in range(3):
            mask = (labels_for_plot == c)
            ax.scatter(X2[mask, 0], X2[mask, 1], X2[mask, 2],
                    s=40, c=colors[c], label=f"Cluster {c}")

        ax.set_xlabel("pr_k_scaled")
        ax.set_ylabel("score_scaled")
        ax.set_zlabel("test_score_scaled")
        ax.legend()
        plt.show()


if __name__ == "__main__":
    main()
