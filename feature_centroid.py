import sys
import os
import numpy as np
import csv
from tqdm import tqdm
from file_util import get_files, open_bin
from multiprocessing import Pool
import argparse
import pandas as pd

def extract_region(points: np.ndarray, region=None):
    if region is None:
        return points
    xmin, xmax, ymin, ymax, zmin, zmax = region
    mask = (
        (points[:,0] >= xmin) & (points[:,0] <= xmax) &
        (points[:,1] >= ymin) & (points[:,1] <= ymax) &
        (points[:,2] >= zmin) & (points[:,2] <= zmax)
    )
    return points[mask]

def process_frame_centroid(args):
    """
    1フレーム(binファイル)の点群に対し、重心を計算して返す。
    """
    bf, d, region_3d = args

    # .binファイルを読み込む (x,y,z,...)
    points = open_bin(os.path.join(d, bf))[:, :3]

    # region_3d があればフィルタリング
    points = extract_region(points, region_3d)

    # 点が無い場合は (0,0,0) とする (またはNaNなどでも可)
    if len(points) == 0:
        return (0.0, 0.0, 0.0)

    # (x,y,z)の重心(平均座標)
    x_mean = np.mean(points[:,0])
    y_mean = np.mean(points[:,1])
    z_mean = np.mean(points[:,2])

    return (x_mean, y_mean, z_mean)

def main():
    """
    python3 characterize_centroid.py [merged_dir] [output_csv(optional)]
    
    - 点群(.bin)の各フレームの重心を計算し、
      全フレームの重心から算出する「全体平均重心」までの距離をp50, p90にまとめる。
    - CSV出力時の仕様:
      1. ID列が既に存在する場合は同じ行に上書き
      2. 存在しない場合は新規行を追加
      3. CentroidDist_50%, CentroidDist_90%列が無ければ追加
    """
    parser = argparse.ArgumentParser(description="Compute centroid distribution features from pointcloud frames.")
    parser.add_argument("pcd_dir", help="Directory containing merged pointcloud (.bin) files.")
    parser.add_argument("csv_file", nargs="?", default=None, 
                        help="Output CSV file (optional). If not specified, feature_centroid.csv is created or updated.")
    args = parser.parse_args()

    pcd_dir = args.pcd_dir
    user_csv_file = args.csv_file

    # CSV出力先ディレクトリ
    csv_output_dir = "./pointcloud_feature"
    os.makedirs(csv_output_dir, exist_ok=True)

    # 出力先CSVの決定
    if user_csv_file is None:
        default_csv_path = os.path.join(csv_output_dir, "feature_centroid.csv")
        output_csv = default_csv_path
    else:
        output_csv = user_csv_file

    # 読み込みたい3D領域(必要に応じて)
    region_3d = (-600.0, -130.0, 440.0, 1037.0, 220.0, 540.0)
    # region_3d = None

    # binファイルを取得
    bin_files = sorted(get_files(pcd_dir, "pcd", mode="KITTI"))

    # 各フレームの重心を計算 (並列)
    args_list = [(bf, pcd_dir, region_3d) for bf in bin_files]
    centroids = []
    with Pool() as pool:
        for centroid in tqdm(pool.imap(process_frame_centroid, args_list), total=len(args_list), desc="Computing centroids"):
            centroids.append(centroid)

    if len(centroids) == 0:
        print("No frames found. Exiting...")
        return

    # 全フレーム重心の平均座標を計算
    centroids_np = np.array(centroids)  # shape: (N, 3)
    Cx = np.mean(centroids_np[:,0])
    Cy = np.mean(centroids_np[:,1])
    Cz = np.mean(centroids_np[:,2])

    # 各フレーム重心から全体重心までの距離
    dists = np.sqrt((centroids_np[:,0] - Cx)**2
                    + (centroids_np[:,1] - Cy)**2
                    + (centroids_np[:,2] - Cz)**2)

    # 距離分布から 50% と 90% を取得
    p50 = float(np.percentile(dists, 50)) if len(dists) > 0 else 0.0
    p90 = float(np.percentile(dists, 90)) if len(dists) > 0 else 0.0

    # データセット名を ID とする
    dataset_name = os.path.basename(os.path.normpath(pcd_dir))

    # --- CSVの既存内容を読み込み or 新規作成し、IDをキーにして更新する ---
    if os.path.exists(output_csv):
        # 既存CSVがある場合は読み込み
        df = pd.read_csv(output_csv)
    else:
        # 無ければ空のDataFrameを作成 (後で列を追加していく)
        df = pd.DataFrame()

    # 1) ID列が存在しない場合は追加
    #    ID列がある場合でも、それをキーとして扱えるようにする
    if 'ID' not in df.columns:
        df['ID'] = []  # 新規列

    # 2) 必要な列(CentroidDist_50%, CentroidDist_90%)が無ければ追加
    if 'CentroidDist_50%' not in df.columns:
        df['CentroidDist_50%'] = np.nan
    if 'CentroidDist_90%' not in df.columns:
        df['CentroidDist_90%'] = np.nan

    # DataFrameを ID で検索・更新しやすいように index を ID にする
    # ※ 既に重複したIDがあると想定した場合は要検討 (通常は1行想定)
    df.set_index('ID', inplace=True)

    # 3) dataset_name が既に存在するかをチェック → 上書き or 新規行
    df.loc[dataset_name, 'CentroidDist_50%'] = p50
    df.loc[dataset_name, 'CentroidDist_90%'] = p90

    # index を戻して CSVへ書き込み
    df.reset_index(inplace=True)
    df.to_csv(output_csv, index=False)

    print(f"ID='{dataset_name}' の CentroidDist_50%, 90% を CSV に書き込みました: {output_csv}")

if __name__ == "__main__":
    main()
