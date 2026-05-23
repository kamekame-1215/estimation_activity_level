import sys
import os
import numpy as np
import json
import csv
from tqdm import tqdm
from file_util import get_files, open_bin
from multiprocessing import Pool
import argparse
import matplotlib.pyplot as plt

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

def project_pointcloud_to_image(points: np.ndarray, axes=('x','y'), resolution=(100, 100), region=None):
    """
    points: Nx3
    axes: ('x','y'), ('y','z'), ('z','x') のいずれかを指定
        'x','y','z'はpoints[:,0], points[:,1], points[:,2]に対応
    resolution: 画像解像度(H,W)
    region: (min_axis1, max_axis1, min_axis2, max_axis2)
            axesに対応する2D投影空間での範囲指定

    戻り値: HxWの画像 (np.float64)
    """
    if points.shape[0] == 0:
        return np.zeros(resolution, dtype=np.float64)

    # 軸選択に応じて対応する座標を取り出す
    coord_map = {'x':0, 'y':1, 'z':2}
    ax1, ax2 = axes
    p1 = points[:, coord_map[ax1]]
    p2 = points[:, coord_map[ax2]]

    if region is None:
        min_p1, max_p1 = np.min(p1), np.max(p1)
        min_p2, max_p2 = np.min(p2), np.max(p2)
    else:
        min_p1, max_p1, min_p2, max_p2 = region

    H, W = resolution
    image = np.zeros((H, W), dtype=np.float64)

    p1_range = max_p1 - min_p1
    p2_range = max_p2 - min_p2
    if p1_range == 0 or p2_range == 0:
        return image

    # p1 -> width方向, p2 -> height方向 という前提
    # p2は高さ方向だが、画像は上->下に行indexが増えるので注意(ここでは反転している)
    p1_pix = ((p1 - min_p1) / p1_range) * (W - 1)
    p2_pix = ((p2 - min_p2) / p2_range) * (H - 1)
    p1_pix_int = np.floor(p1_pix).astype(int)
    p2_pix_int = np.floor(p2_pix).astype(int)

    mask = (p1_pix_int >= 0) & (p1_pix_int < W) & (p2_pix_int >= 0) & (p2_pix_int < H)
    p1_pix_int = p1_pix_int[mask]
    p2_pix_int = p2_pix_int[mask]

    for yp, xp in zip(p2_pix_int, p1_pix_int):
        image[H - 1 - yp, xp] += 1.0

    return image

def compute_zncc(img1: np.ndarray, img2: np.ndarray):
    mean1 = np.mean(img1)
    mean2 = np.mean(img2)
    num = np.sum((img1 - mean1) * (img2 - mean2))
    den = np.sqrt(np.sum((img1 - mean1)**2)*np.sum((img2 - mean2)**2))
    if den == 0:
        return 0.0
    return num / den

def process_frame(args):
    """
    args: (bf, d, region_3d, projection_region, resolution)

    この関数は1フレーム分の点群を読み込んで、3パターンの投影画像を返すように変更。
    戻り値は (img_xy, img_yz, img_zx) のタプルとする。
    """
    bf, d, region_3d, projection_region, resolution = args
    points = open_bin(os.path.join(d, bf))[:, :3]
    points = extract_region(points, region_3d)

    # 3種類の投影を実施
    img_xy = project_pointcloud_to_image(points, axes=('x','y'), resolution=resolution, region=None)
    img_yz = project_pointcloud_to_image(points, axes=('y','z'), resolution=resolution, region=None)
    img_zx = project_pointcloud_to_image(points, axes=('z','x'), resolution=resolution, region=None)

    return (img_xy, img_yz, img_zx)

def main():
    """
    【プログラム実行時の引数例】
    python3 characterize_zncc.py [merged_dir] [output_csv(optional)]

    [merged_dir]: マージ後点群(bin)が格納されたディレクトリ(一つ)
    [output_csv]: 結果を出力するCSVファイルのパス(省略可)
                  省略された場合は feature_ZNCC.csv を新規作成
                  指定された場合は既存ファイルに追記、なければ新規作成

    フレーム画像の確認は、コード内のsave_first_imageフラグで制御
    今回は3軸分計算するので、出力は以下:
    ID, ZNCC_50_xy, ZNCC_90_xy, ZNCC_50_yz, ZNCC_90_yz, ZNCC_50_zx, ZNCC_90_zx
    """

    parser = argparse.ArgumentParser(description="Compute ZNCC-based similarity features between consecutive frames on 3 projections.")
    parser.add_argument("pcd_dir", help="Directory containing merged pointcloud files.")
    parser.add_argument("csv_file", nargs="?", default=None, help="Output CSV file (optional). If not specified, feature_ZNCC.csv is created.")
    args = parser.parse_args()

    pcd_dir = args.pcd_dir
    user_csv_file = args.csv_file

    # CSV出力先ディレクトリ(プログラム内で指定可能)
    csv_output_dir = "./pointcloud_feature"
    os.makedirs(csv_output_dir, exist_ok=True)

    if user_csv_file is None:
        # 指定なしの場合 feature_ZNCC.csvを新規または追記
        default_csv_path = os.path.join(csv_output_dir, "feature_ZNCC.csv")
        if os.path.exists(default_csv_path):
            write_mode = 'a'
            header_needed = False
        else:
            write_mode = 'w'
            header_needed = True
        output_csv = default_csv_path
    else:
        output_csv = user_csv_file
        if os.path.exists(output_csv):
            write_mode = 'a'
            header_needed = False
        else:
            write_mode = 'w'
            header_needed = True

    # region_3d = None
    region_3d = (-575,120,714,1384,-427,472) #(xmin, xmax, ymin, ymax, zmin, zmax)
    projection_region = None
    resolution = (100, 100) #解像度の設定

    # フレーム画像確認フラグ
    save_first_image = True

    dataset_name = os.path.basename(os.path.normpath(pcd_dir))
    bin_files = get_files(pcd_dir, "pcd", mode="KITTI")
    bin_files = sorted(bin_files)

    args_list = [(bf, pcd_dir, region_3d, projection_region, resolution) for bf in bin_files]

    # images_xy, images_yz, images_zx に分けて格納する
    images_xy = []
    images_yz = []
    images_zx = []
    with Pool() as pool:
        for img_tuple in tqdm(pool.imap(process_frame, args_list), total=len(args_list), desc=f"Processing {dataset_name}"):
            img_xy, img_yz, img_zx = img_tuple
            images_xy.append(img_xy)
            images_yz.append(img_yz)
            images_zx.append(img_zx)

    if save_first_image and len(images_xy) > 0:
        # XY投影の最初のフレーム画像を確認として保存(他軸も必要ならコメントアウトを外して保存可能)
        plt.imshow(images_xy[0], cmap='gray')
        plt.title(f"{dataset_name} first frame (XY projection)")
        plt.colorbar()
        plt.savefig(os.path.join(csv_output_dir, f"first_frame_xy_{dataset_name}.png"))
        plt.close()
        # 他軸
        # plt.imshow(images_yz[0], cmap='gray')
        # plt.title(f"{dataset_name} first frame (YZ projection)")
        # plt.colorbar()
        # plt.savefig(os.path.join(csv_output_dir, "first_frame_yz.png"))
        # plt.close()

        # plt.imshow(images_zx[0], cmap='gray')
        # plt.title(f"{dataset_name} first frame (ZX projection)")
        # plt.colorbar()
        # plt.savefig(os.path.join(csv_output_dir, "first_frame_zx.png"))
        # plt.close()

    # 各軸ペアごとにZNCC計算
    def compute_zncc_stats(images):
        zncc_values = []
        for i in tqdm(range(len(images)-1), desc="Computing ZNCC", leave=True):
            zncc_val = compute_zncc(images[i], images[i+1])
            zncc_values.append(zncc_val)
        if len(zncc_values) == 0:
            return 0.0, 0.0
        else:
            p50 = float(np.percentile(zncc_values, 50))
            p90 = float(np.percentile(zncc_values, 90))
            return p50, p90

    p50_xy, p90_xy = compute_zncc_stats(images_xy)
    p50_yz, p90_yz = compute_zncc_stats(images_yz)
    p50_zx, p90_zx = compute_zncc_stats(images_zx)

    # CSV出力
    with open(output_csv, write_mode, newline='') as f:
        writer = csv.writer(f)
        if header_needed:
            writer.writerow(["ID", "ZNCC_50_xy", "ZNCC_90_xy", "ZNCC_50_yz", "ZNCC_90_yz", "ZNCC_50_zx", "ZNCC_90_zx"])
        writer.writerow([dataset_name, p50_xy, p90_xy, p50_yz, p90_yz, p50_zx, p90_zx])

    print("特徴量抽出が完了し、結果をCSVに出力しました:", output_csv)


if __name__ == "__main__":
    main()
