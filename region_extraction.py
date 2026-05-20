from typing import List, Tuple
from file_util import write_bin, load_bin, get_files

import numpy as np
import sys
import os
import open3d as o3d
from multiprocessing import Pool
from tqdm import tqdm
import argparse

def extract_area(point_cloud: np.ndarray) -> np.ndarray:
    """
    点群データから指定された領域内の点を抽出する関数。q
    """
    x_coords: np.ndarray = point_cloud[:, 0]
    y_coords: np.ndarray = point_cloud[:, 1]

    # 領域の条件を設定
    #設定1(hirai)
    # condition: np.ndarray = (
    #     (x_coords > -300) & (x_coords < 700) &
    #     (y_coords > 700) & (y_coords < 1600)
    # )
    #設定2(kudo)
    condition: np.ndarray = (
        (x_coords > -600) & (x_coords < 100) &
        (y_coords > 400) & (y_coords < 1250)
    )
    filtered_points: np.ndarray = point_cloud[condition]

    return filtered_points

def remove_plane(point_cloud_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[int], List[float]] :
    """
    点群データから平面を推定し、平面を除去した点群を返す関数。
    """
    # NumPyの配列をOpen3DのPointCloudに変換
    point_cloud_o3d = o3d.geometry.PointCloud()
    point_cloud_o3d.points = o3d.utility.Vector3dVector(point_cloud_np)

    # RANSAC
    plane_model, inliers = point_cloud_o3d.segment_plane(
        distance_threshold=11.1, #距離閾値
        ransac_n=3, #サンプル数(平面はn=3)
        num_iterations=5000 #試行回数
    )

    # 平面の方程式の係数
    [a, b, c, d] = plane_model
    # print(f"Plane equation: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")

    # 平面に属さない点を抽出
    point_cloud_without_plane = point_cloud_o3d.select_by_index(inliers, invert=True)
    filtered_points: np.ndarray = np.asarray(point_cloud_without_plane.points)

    return filtered_points, inliers, plane_model

def visualize_result(point_cloud_np: np.ndarray, inliers: List[int], plane_model: List[float]) -> None:
    """
    点群データと推定された平面を視覚的に表示する関数。
    """
    # Open3DのPointCloudオブジェクトに変換
    point_cloud_o3d = o3d.geometry.PointCloud()
    point_cloud_o3d.points = o3d.utility.Vector3dVector(point_cloud_np)

    # インライアとアウトライアを分離
    inlier_cloud = point_cloud_o3d.select_by_index(inliers)
    outlier_cloud = point_cloud_o3d.select_by_index(inliers, invert=True)

    # 色を設定
    inlier_cloud.paint_uniform_color([1.0, 0, 0])    # 赤色
    outlier_cloud.paint_uniform_color([0, 1.0, 0])   # 緑色

    # 平面を作成
    plane_mesh = create_plane_mesh(plane_model, point_cloud_np)

    # 可視化
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud, plane_mesh])

    return None

def create_plane_mesh(plane_model: List[float], point_cloud_np: np.ndarray) -> o3d.geometry.TriangleMesh:
    """
    推定された平面を三角メッシュとして作成する関数。
    """
    [a, b, c, d] = plane_model

    # 点群の中心を計算
    centroid = point_cloud_np.mean(axis=0)

    # 平面の法線ベクトル
    normal = np.array([a, b, c])

    # 平面上の4つの頂点を計算
    plane_size = 10  # 平面のサイズを調整
    orthogonal_vector = np.array([-b, a, 0])  # 法線に直交するベクトル

    if np.linalg.norm(orthogonal_vector) == 0:
        orthogonal_vector = np.array([0, -c, b])

    orthogonal_vector /= np.linalg.norm(orthogonal_vector)
    vec1 = np.cross(normal, orthogonal_vector)
    vec2 = np.cross(normal, vec1)

    vec1 *= plane_size
    vec2 *= plane_size

    corners = [
        centroid + vec1 + vec2,
        centroid + vec1 - vec2,
        centroid - vec1 - vec2,
        centroid - vec1 + vec2,
    ]

    # 三角形の頂点を定義
    vertices = o3d.utility.Vector3dVector(corners)
    triangles = o3d.utility.Vector3iVector([[0, 1, 2], [2, 3, 0]])

    # メッシュを作成
    plane_mesh = o3d.geometry.TriangleMesh(vertices, triangles)
    plane_mesh.compute_vertex_normals()
    plane_mesh.paint_uniform_color([0, 0, 1.0])  # 青色

    return plane_mesh

def process_file(args: Tuple[str, str, str, bool]) -> None:
    """
    単一のファイルを処理する関数：
    点群データの読み込み、領域抽出、平面推定と除去、保存、必要に応じて視覚化を行う。
    """
    file_name: str
    input_dir: str
    output_dir: str
    visualize: bool
    file_name, input_dir, output_dir, visualize = args

    # print(f"Processing {file_name}")

    # 入力ファイルのパスを生成
    input_file_path: str = os.path.join(input_dir, file_name)

    # 点群データの読み込み
    point_cloud_np: np.ndarray = load_bin(input_file_path)

    # 領域抽出
    extracted_point_cloud: np.ndarray = extract_area(point_cloud_np)

    # 平面推定と除去
    filtered_point_cloud, inliers, plane_model = remove_plane(extracted_point_cloud)

    # ファイル名の取得とログ出力
    base_name: str = os.path.splitext(os.path.basename(file_name))[0]
    # print(f"Processing {base_name}")

    # 出力ファイルのパスを生成
    output_file_path: str = os.path.join(output_dir, base_name + "_processed.bin")

    # 処理した点群データを保存
    write_bin(output_file_path, filtered_point_cloud, has_intensity=False)

    # 視覚化（試験的な実行時のみ）
    if visualize:
        visualize_result(extracted_point_cloud, inliers, plane_model)

def main() -> None:
    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(description="Point Cloud Processing")
    parser.add_argument("input_dir", help="Input directory containing point cloud files")
    parser.add_argument("output_dir", help="Output directory to save processed files")
    parser.add_argument("--test", "-t", action="store_true", help="Enable test mode (process only one file with visualization)")
    args = parser.parse_args()

    input_dir: str = args.input_dir
    output_dir: str = args.output_dir
    test_mode: bool = args.test  # コマンドライン引数で設定

    # 出力ディレクトリが存在しない場合は新規作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 入力ディレクトリ内の .pcd ファイルを取得
    bin_files: List[str] = get_files(input_dir, "pcd", mode="LiDAR")

    if not bin_files:
        print("No .bin files found in the input directory.")
        return

    if test_mode:
        # 試験的に最初の1ファイルのみを処理
        first_file = bin_files[0]
        process_file((first_file, input_dir, output_dir, True))
    else:
        # 全てのファイルを並列処理
        visualize = False  # 全体処理時は視覚化しない
        args_list: List[Tuple[str, str, str, bool]] = [(file_name, input_dir, output_dir, visualize) for file_name in bin_files]

        with Pool() as pool:
            list(tqdm(pool.imap_unordered(process_file, args_list), total=len(args_list)))


if __name__ == '__main__':
    main()