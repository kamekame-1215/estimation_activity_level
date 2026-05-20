import open3d as o3d
import time
import cv2
import threading

from sys import argv

from file_util import open_bin, get_files
from pcd_util import np_pcd
from pcd_util import *

"""
フォルダ内の点群データをコマ送りで見る

使い方
python3 frame_viewer.py binディレクトリパス ファイル名の種類 [初期の表示フレーム番号]

ファイル名の種類は
127.0.0.1-9010_1_1662439120642232.binのような形式なら LiDAR
000001.binのような形式なら KITTI
を実行時引数に

1キー表示するフレームを1進める
2キー表示するフレームを10進める
9キー表示するフレームを1戻す
0キー表示するフレームを10戻す
Hキーでその他操作説明

"""   


def frame_viewer(dirpass: str, file_kind: str, start_frame:int=0) -> None:
    files = get_files(dirpass, "pcd", file_kind)
    frame = start_frame
    file_num = len(files)
    pcd = o3d.geometry.PointCloud()
    pcd_add = o3d.geometry.PointCloud()
    points = open_bin(dirpass + "/" + files[frame])
    points = points[:, :3]
    points_add = points
    #points_line = points_line[:, :3]
    #外れ値をけす
    points = points[np.all((-100000.0 < points) & (points < 100000.0), axis=1)] #デフォルト
    # points = points[np.all((-2000.0 < points) & (points < 2000.0), axis=1)] # 追加実験全体確認用
    
    x = points[:, 0]
    y = points[:, 1] 
    z = points[:, 2]
    # points = points[np.where((y<1600))] 

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.3, 0.3, 0.3])

    xy_plane = create_xy_plane_LineSet(0.2, 200, -1.70)
    xy_plane = create_xy_plane_LineSet(1, 100, -1.730)
    yz_plane_max = create_yz_plnae_LineSet(1, 50, 69.120)
    yz_plane_min = create_yz_plnae_LineSet(1, 50, 0.0)
    zx_plane_max = create_zx_plnae_LineSet(1, 50, 39.680)
    zx_plane_min = create_zx_plnae_LineSet(1, 50, -39.680)
    #xy_plane = create_xy_plane_LineSet(100, 100, -1730)
    #yz_plane_max = create_yz_plnae_LineSet(100, 50, 69120)
    #yz_plane_min = create_yz_plnae_LineSet(100, 50, 00)
    #zx_plane_max = create_zx_plnae_LineSet(100, 50, 39680)
    #zx_plane_min = create_zx_plnae_LineSet(100, 50, -39680)

    #plane = create_plnae_LineSet(1, 10, 10)
    

    def update_pcd():
        nonlocal pcd, pcd_add, files, frame, dirpass
        points = open_bin(dirpass + "/" + files[frame])
        points = points[:, :3]
        print(points.dtype)
        # points[:, 0] = (points[:,0] >= -575.82) & (points[:,0] <= -100.31)
        # points[:, 1] = (points[:,1] >= 714.41) & (points[:,1] <= 1384.81)
        # points[:, 2] = (points[:,2] >= -427.47) & (points[:,2] <= 472.61)
        #外れ値をけす
        #points = points[points[:,0] >= points[:,1]*5 - 27000]
        #points = points[np.all((-100000.0 < points) & (points < 100000.0), axis=1)]
        # points = points[points[:, 2] > 100]
        # points = points[points[:, 2] < 2000]
        # points = points[points[:, 2] > -10000]
        # points = points[points[:, 2] < 200000]

        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color([0.3, 0.3, 0.3])
        print(files[frame], frame)

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

    # def advance_frames():
    #     while frame < file_num - 1:
    #         plus1_frame()
    #         time.sleep(0.5)
    #     return True
    
    # def start_advance_frames(vis):
    #     threading.Thread(target=advance_frames).start()
    #     return True

    # is_recording = False
    # video_writer = None

    # def start_recording(vis):
    #     nonlocal is_recording, video_writer, frame
    #     is_recording = True
    #     fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #     video_writer = cv2.VideoWriter('recording.avi', fourcc, 5.0, (640, 480))
    #     frame = 0
    #     while frame < file_num:
    #         update_pcd()
    #         vis.capture_screen_image('frame.jpg')
    #         img = cv2.imread('frame.jpg')
    #         video_writer.write(img)
    #         frame += 1
    #         time.sleep(0.2)
    #     video_writer.release()
    #     is_recording = False
    #     return True

    callback_list ={ord("1"):plus1_frame, 
                    ord("2"):plus10_frame,
                    ord("9"):minus1_frame, 
                    ord("0"):minus10_frame
                    # ord("3"): start_advance_frames
                    # ord("s"): start_recording
                    }

    #o3d.visualization.draw_geometries_with_key_callbacks([pcd, xy_plane, yz_plane_max, yz_plane_min, zx_plane_max, zx_plane_min], key_to_callback=callback_list)
    o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback=callback_list)

if __name__ == "__main__":
    if len(argv) == 4:
        start_frame = int(argv[3])
    else :
        start_frame = 0
    
    frame_viewer(argv[1], argv[2], start_frame)