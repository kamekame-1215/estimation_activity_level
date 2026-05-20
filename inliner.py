from typing import List, Tuple
from file_util import write_bin, get_files
import numpy as np
import sys
import os
import open3d as o3d

def load_point_cloud(file_path: str) -> np.ndarray:
    """
    バイナリファイルから点群データを読み込む関数。
    """
    with open(file_path, 'rb') as f:
        point_cloud: np.ndarray = np.fromfile(f, dtype=np.float32)
    point_cloud = point_cloud.reshape(-1, 3)  # (N, 3)の形状に変換
    return point_cloud

def remove_noise(point_cloud_np: np.ndarray) -> np.ndarray:
    """
    点群データからノイズを除去する関数。
    統計的外れ値除去を使用。
    """
    # NumPy配列をOpen3DのPointCloudオブジェクトに変換
    point_cloud_o3d = o3d.geometry.PointCloud()
    point_cloud_o3d.points = o3d.utility.Vector3dVector(point_cloud_np)
    
    # 統計的外れ値除去
    cl, ind = point_cloud_o3d.remove_statistical_outlier(
        nb_neighbors=20,  # 近傍点の数
        std_ratio=2.0     # 標準偏差の倍数
    )
    
    # ノイズを除去した点群データを取得
    filtered_point_cloud = point_cloud_o3d.select_by_index(ind)
    
    # NumPy配列に変換して返す
    filtered_points: np.ndarray = np.asarray(filtered_point_cloud.points)
    return filtered_points

def estimate_plane(point_cloud_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
    """
    点群データから平面を推定し、平面に属する点と属さない点を返す関数。
    また、平面の方程式の係数を返す。
    """
    # NumPyの配列をOpen3DのPointCloudに変換
    point_cloud_o3d = o3d.geometry.PointCloud()
    point_cloud_o3d.points = o3d.utility.Vector3dVector(point_cloud_np)

    # RANSAC法で平面を検出
    plane_model, inliers = point_cloud_o3d.segment_plane(
        distance_threshold=10.0, #距離閾値
        ransac_n=3, #サンプル数(平面はn=3)
        num_iterations=5000 #試行回数
    )

    # 平面の方程式の係数
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")

    # インライアとアウトライアのインデックス
    inlier_cloud = point_cloud_o3d.select_by_index(inliers)
    outlier_cloud = point_cloud_o3d.select_by_index(inliers, invert=True)

    # NumPy配列に変換
    inlier_points: np.ndarray = np.asarray(inlier_cloud.points)
    outlier_points: np.ndarray = np.asarray(outlier_cloud.points)

    return inlier_points, outlier_points, inliers, plane_model

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

def process_single_file(file_name: str, input_dir: str, output_dir: str) -> None:
    """
    単一のファイルを処理する関数：
    点群データの読み込み、平面推定と除去、保存、視覚化を行う。
    """
    # 入力ファイルのパスを生成
    input_file_path: str = os.path.join(input_dir, file_name)

    # 点群データの読み込み
    point_cloud_np: np.ndarray = load_point_cloud(input_file_path)

    # ノイズ除去
    # denoised_point_cloud_np: np.ndarray = remove_noise(point_cloud_np)

    # 平面推定とインライア・アウトライアの取得
    inlier_points, outlier_points, inliers, plane_model = estimate_plane(point_cloud_np)

    # ファイル名の取得とログ出力
    base_name: str = os.path.splitext(os.path.basename(file_name))[0]
    print(f"Processing {base_name}")

    # 出力ファイルのパスを生成
    output_file_path: str = os.path.join(output_dir, base_name + "_no_plane.bin")

    # 平面を除去した点群データを保存
    write_bin(output_file_path, outlier_points, True)

    # 視覚化
    visualize_result(point_cloud_np, inliers, plane_model)

def main() -> None:
    # コマンドライン引数から入力ディレクトリと出力ディレクトリのパスを取得
    input_dir: str = sys.argv[1]
    output_dir: str = sys.argv[2]

    # 出力ディレクトリが存在しない場合は新規作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 入力ディレクトリ内の .pcd ファイルを取得
    bin_files: List[str] = get_files(input_dir, "pcd", mode="LiDAR")

    if not bin_files:
        print("No .pcd files found in the input directory.")
        return

    # 最初の1ファイルのみを処理
    first_file = bin_files[0]
    process_single_file(first_file, input_dir, output_dir)

if __name__ == '__main__':
    main()
