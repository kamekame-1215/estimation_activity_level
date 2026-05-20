import open3d as o3d
import time
import cv2
import os
import threading
import numpy as np
from sys import argv

from file_util import open_bin, get_files
from pcd_util import np_pcd
from pcd_util import *

"""
フォルダ内の点群データをコマ送りで見る

使い方:
python3 frame_viewer.py binディレクトリパス ファイル名の種類 [初期の表示フレーム番号]

キー操作:
1 : フレームを1進める
2 : フレームを10進める
9 : フレームを1戻す
0 : フレームを10戻す
6 : 領域(xmin,xmax,ymin,ymax,zmin,zmax)をコンソールから入力して更新
    qと入力するとキャンセル
"""

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

def create_bounding_box_lineset(region):
    """
    region = (xmin, xmax, ymin, ymax, zmin, zmax) を頂点とする直方体の辺をLineSetで返す
    """
    xmin, xmax, ymin, ymax, zmin, zmax = region
    # 頂点座標
    corners = np.array([
        [xmin, ymin, zmin],
        [xmin, ymin, zmax],
        [xmin, ymax, zmin],
        [xmin, ymax, zmax],
        [xmax, ymin, zmin],
        [xmax, ymin, zmax],
        [xmax, ymax, zmin],
        [xmax, ymax, zmax]
    ], dtype=np.float64)

    # 辺を結ぶインデックス（8頂点直方体の12エッジ）
    edges = np.array([
        [0,1],[0,2],[0,4],
        [1,3],[1,5],
        [2,3],[2,6],
        [3,7],
        [4,5],[4,6],
        [5,7],
        [6,7]
    ], dtype=np.int32)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(edges)
    # バウンディングボックスは目立つ色(赤)にする
    line_set.colors = o3d.utility.Vector3dVector(np.tile([1.0,0,0],[edges.shape[0],1]))
    return line_set

def frame_viewer(dirpass: str, file_kind: str, start_frame:int=0) -> None:
    files = get_files(dirpass, "pcd", file_kind)
    frame = start_frame
    file_num = len(files)
    pcd = o3d.geometry.PointCloud()
    region = None  # 領域指定用変数
    bounding_box = None  # 領域表示用LineSet
    current_points_range = None  # 現在表示中の点群範囲を記録 (xmin,xmax,ymin,ymax,zmin,zmax)

    def update_pcd():
        nonlocal pcd, files, frame, dirpass, region, current_points_range
        points = open_bin(os.path.join(dirpass, files[frame]))[:, :3]
        filtered_points = extract_region(points, region)
        pcd.points = o3d.utility.Vector3dVector(filtered_points)
        pcd.paint_uniform_color([0.3, 0.3, 0.3])

        if len(filtered_points) > 0:
            xmin = np.min(filtered_points[:,0])
            xmax = np.max(filtered_points[:,0])
            ymin = np.min(filtered_points[:,1])
            ymax = np.max(filtered_points[:,1])
            zmin = np.min(filtered_points[:,2])
            zmax = np.max(filtered_points[:,2])
            current_points_range = (xmin,xmax,ymin,ymax,zmin,zmax)
        else:
            # フィルタ後点がなければ範囲不明
            current_points_range = None

        print("表示中ファイル:", files[frame], "フレーム:", frame, "点数:", len(filtered_points))
        if current_points_range is not None:
            print("現在表示中点群の範囲:")
            print("X:[{:.2f}, {:.2f}] Y:[{:.2f}, {:.2f}] Z:[{:.2f}, {:.2f}]".format(*current_points_range))
        else:
            print("現在表示中点群なし")
        return True

    def update_region_box(vis):
        nonlocal region, bounding_box
        # 既存のbounding_boxを削除
        if bounding_box is not None:
            vis.remove_geometry(bounding_box, reset_bounding_box=False)
            bounding_box = None
        
        # regionが設定されていれば新たにボックスを追加
        if region is not None:
            new_box = create_bounding_box_lineset(region)
            vis.add_geometry(new_box, reset_bounding_box=False)
            bounding_box = new_box
        return True

    def plus10_frame(vis):
        nonlocal file_num, frame
        frame = min(frame+10, file_num-1)
        update_pcd()
        return True
    
    def plus1_frame(vis):
        nonlocal file_num, frame
        frame = min(frame+1, file_num-1)
        update_pcd()
        return True

    def minus10_frame(vis):
        nonlocal file_num, frame
        frame = max(frame-10, 0)
        update_pcd()
        return True
    
    def minus1_frame(vis):
        nonlocal file_num, frame
        frame = max(frame-1, 0)
        update_pcd()
        return True

    def adjust_region(vis):
        nonlocal region, current_points_range
        print("現在の領域:", region)
        if current_points_range is not None:
            print("現在表示中点群の座標範囲:")
            print("X:[{:.2f}, {:.2f}] Y:[{:.2f}, {:.2f}] Z:[{:.2f}, {:.2f}]".format(*current_points_range))
        else:
            print("現在表示中の点群は存在しません。(領域が狭すぎる可能性があります)")

        print("xmin,xmax,ymin,ymax,zmin,zmax の形式で入力してください。例: -10,10,-5,5,0,20")
        print("qと入力すると領域指定をキャンセルします。")
        line = input("領域を指定してください: ")
        line = line.strip()
        if line.lower() == 'q':
            print("領域指定をキャンセルしました。")
            return True
        try:
            vals = line.split(",")
            if len(vals) != 6:
                print("6つの値をカンマ区切りで入力してください。")
                return True
            xmin, xmax, ymin, ymax, zmin, zmax = map(float, vals)
            region = (xmin, xmax, ymin, ymax, zmin, zmax)
            print("新しい領域:", region)
            update_pcd()
            # 領域更新したのでバウンディングボックス表示を更新
            update_region_box(vis)
        except ValueError:
            print("数値に変換できませんでした。再度正しく入力してください。")
        return True

    # 初期表示更新
    update_pcd()

    callback_list = {
        ord("1"): plus1_frame, 
        ord("2"): plus10_frame,
        ord("9"): minus1_frame, 
        ord("0"): minus10_frame,
        ord("6"): adjust_region
    }

    # 初回は領域なしなのでバウンディングボックスなし
    o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback=callback_list)


if __name__ == "__main__":
    if len(argv) == 4:
        start_frame = int(argv[3])
    else:
        start_frame = 0
    
    frame_viewer(argv[1], argv[2], start_frame)
