import os
import time
import numpy as np
import cv2
import open3d as o3d
from sys import argv

from file_util import open_bin, get_files

def frame_viewer(dirpass: str, file_kind: str, start_frame: int = 0) -> None:
    """
    指定されたディレクトリ内の点群データを順に自動で可視化し、
    任意のタイミングで録画を行う関数。

    Args:
        dirpass (str): 点群データが保存されているディレクトリのパス。
        file_kind (str): ファイル名の種類（"LiDAR" または "KITTI"）。
        start_frame (int, optional): 表示を開始するフレーム番号。デフォルトは0。
    """

    def print_help(vis):
        """
        操作方法をコンソールに表示する関数。
        """
        print("""
        操作方法:
        - 自動で点群データが連続再生されます。
        - 'S' キーを押すと、録画の開始・停止を切り替えることができます。
        - 'H' キーを押すと、このヘルプメッセージが表示されます。
        """)
        
        return False

    # 操作方法の表示
    print_help(None)

    files = get_files(dirpass, "pcd", file_kind)
    frame = start_frame
    file_num = len(files)
    pcd = o3d.geometry.PointCloud()

    # 初期点群の読み込み
    points = open_bin(os.path.join(dirpass, files[frame]))
    points = points[:, :3]
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.3, 0.3, 0.3])

    # 録画関連の変数
    is_recording = False
    video_writer = None
    recording_count = 1

    # 入力ディレクトリの末尾のフォルダ名を取得
    dir_name = os.path.basename(os.path.normpath(dirpass))
    # ビデオの保存先（ここで保存先を指定）
    video_save_dir = "videos"  # ビデオファイルの保存先ディレクトリ
    os.makedirs(video_save_dir, exist_ok=True)

    # Visualizerの作成
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.add_geometry(pcd)

    def update_vis():
        """
        Visualizerを更新し、次のフレームの点群データを表示する関数。
        録画中であれば、現在のフレームをビデオに保存する。
        """
        nonlocal pcd, frame, files, dirpass, is_recording, video_writer, file_num
        if frame >= file_num:
            frame = 0  # フレームを最初に戻す
        # 点群データの更新
        points = open_bin(os.path.join(dirpass, files[frame]))
        points = points[:, :3]
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color([0.3, 0.3, 0.3])
        pcd.estimate_normals()
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        if is_recording and video_writer is not None:
            # スクリーンをキャプチャ
            image = vis.capture_screen_float_buffer(False)
            image = np.asarray(image)
            image = (image * 255).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            video_writer.write(image)
        frame += 1  # フレームを進める

    def timer_callback(vis):
        """
        タイマーコールバック関数。
        一定の間隔でVisualizerを更新する。
        """
        update_vis()
        return False  # コールバックを継続

    def start_stop_recording(vis):
        """
        録画の開始・停止を切り替える関数。
        'S' キーが押されたときに呼び出される。
        """
        nonlocal is_recording, video_writer, recording_count
        if not is_recording:
            # 録画開始
            is_recording = True
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            video_filename = f"{dir_name}_{timestamp}_{recording_count}.avi"
            video_path = os.path.join(video_save_dir, video_filename)
            recording_count += 1
            print(f"録画開始: {video_path}")
            # ウィンドウサイズの取得
            width = vis.get_view_control().convert_to_pinhole_camera_parameters().intrinsic.width
            height = vis.get_view_control().convert_to_pinhole_camera_parameters().intrinsic.height
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (width, height))
        else:
            # 録画停止
            is_recording = False
            video_writer.release()
            video_writer = None
            print("録画停止")
        return False

    # キーコールバックの登録
    vis.register_key_callback(ord("S"), start_stop_recording)
    vis.register_key_callback(ord("H"), print_help)

    # タイマーコールバックでフレームを自動更新
    vis.register_animation_callback(timer_callback)

    # Visualizerの実行
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    if len(argv) == 4:
        start_frame = int(argv[3])
    else:
        start_frame = 0
    frame_viewer(argv[1], argv[2], start_frame)
