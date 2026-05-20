import os
import re
from typing import List

import numpy as np
import open3d as o3d

from KITTI_util import KITTILabel, KITTICalib
from pcd_util import *

def write_bin(file_pass: str, points:np.ndarray, has_intensity:bool=True) -> None:
    if not has_intensity:
        points = add_intensity(points)

    with open (file_pass, "wb") as f:
        points = points.ravel()
        for i in range(points.size):
            f.write(points[i].astype(np.float32))

    return None

def load_bin(file_path: str) -> np.ndarray:
    """
    バイナリファイルから点群データを読み込む関数。
    """
    with open(file_path, 'rb') as f:
        point_cloud: np.ndarray = np.fromfile(f, dtype=np.float32)
    point_cloud = point_cloud.reshape(-1, 3)  # (N, 3)の形状に変換
    return point_cloud

def get_files(dirpass: str, file_extension: str, mode:str="LiDAR") -> List[str]:
    """
    点群データのファイルのリストを取得

    Args:
        dirpass: 点群データのディレクトリパス

    Returns:
        List[str]:点群データのファイルのリスト
    """
    files = os.listdir(dirpass)

    if ".DS_Store" in files:
        files.remove(".DS_Store")
        print("ignore .DS_Store")
    print(len(files), "files")
    if mode == "LiDAR":
        p = re.compile(r"(?<=_)\d+(?=_)")
        files.sort(key=lambda s: int(re.search(p, s).group()))
    elif mode == "KITTI":
        p = re.compile(r"\d+(?=.{})".format(file_extension))
        files.sort(key=lambda s: int(re.search(p, s).group()))  # 時刻順でソート
    else :
        print("warning: mode =", mode)
    """
    for s in files:
        if not re.search(p, s):
            #print(s)
            files.remove(s)
    """
    #print(files)
    return files

def open_bin(file_pass: str) -> np.ndarray:
    """
    pcdファイルの読み込み

    Args:
        file_pass: pcdファイルのパス

    Returns:
        np.ndarray: numpy配列 (N, 3)
    """
    pcd = o3d.io.read_point_cloud(file_pass)
    points = np.asarray(pcd.points)
    return points

# def open_KITTI_label(file: str, skip_DontCare:bool=True, scale:float=1.0) -> List:
#     """
#     KITTI形式のラベルを読み込み

#     Returns:
#         ラベル
#     """
#     label = []
#     car_num = 0
#     with open(file) as f:
#         for line in f.readlines():
#             line = line.split(" ")
#             if not skip_DontCare or line[0] != "DontCare":
#                 label.append(KITTILabel.create_from_text(line, scale=scale))
#             if line[0] == "Car":
#                 car_num += 1
    
#     #print("Car num =", car_num)

#     #print(label)
#     print(label)
#     return label


# def open_KITTI_car_label(file: str) -> List:
#     """
#     KITTI形式のラベルを読み込み

#     Returns:
#         ラベル
#     """
#     print("call")
#     label = []
#     r = 0.0
#     with open(file) as f:
#         for line in f.readlines():
#             line = line.split(" ")
#             if line[0] == "Car":
#                 nr = float(line[15])
#                 print(nr)
#                 if nr > r:
#                     label = []
#                     label.append(KITTILabel.create_from_text(line))
#     #print(label)
#     return label



# def open_KITTI_calib(file: str) -> KITTICalib:
#     """
#     KITTI形式のcalibファイルを読み込み

#     Returns:
#         calib
#     """
#     with open(file) as f:
#         lines = f.readlines()
#     return KITTICalib(lines)





# """
# [ [{フレーム番号：ファイル名}, {フレーム番号：ファイル名}], [{フレーム番号：ファイル名}, {フレーム番号：ファイル名}] ]
# """
# def make_files_dict_list(files_list, dirpass:str="") :
#     files_dict_list = []
#     for files in files_list :
#         files_dict = {}
#         for file in files :
#             frame_number = int(file.split("_")[1])
#             files_dict[frame_number] = dirpass+file
#         files_dict_list.append(files_dict)
#     return files_dict_list


# """
# [{フレーム番号：ファイル名}, {フレーム番号：ファイル名}]
# """
# def make_frmae_file_dict(files, dirpass:str="") :

#     files_dict = {}
#     for file in files :
#         frame_number = int(file.split("_")[1])
#         files_dict[frame_number] = dirpass+file
#     return files_dict

# """
# return 最小のフレーム番号, 最大のフレーム番号
# """
# def get_frame_range(files:List[str]):
#     return int(files[0].split("_")[1]), int(files[-1].split("_")[1])


