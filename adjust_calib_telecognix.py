import math
import numpy as np
import open3d as o3d 
from file_util import get_files, np_pcd, open_bin
from pcd_util import *
import sys

from enum import Enum

"""
２つのbinファイルの点群を移動させて変換行列を得る

二つの点群の地面をXY平面にそれぞれ合わせてからキャリブレーションをするとやりやすいと思います


使い方
python3 adjust_calib_3params.py bin1ディレクトリパス 開始フレーム番号1 binディレクトリパス 開始フレーム番号2 [スケール]

引数
binディレクトリパス：bin形式の点群データが入ったフォルダへのパス
開始フレーム番号：フォルダごとに時刻のずれがある場合のための引数。キャリブレーションするために可視化するデータの番号（フォルダ内を昇順にしたとき0から始まる番号）を指定する。時刻がずれて無ければ0と0で良い。
スケール：デフォルトが1。Nを指定すると平行移動するとき移動する距離がN倍になる。

モードごとに出来る動作
PCD1：1つ目の点群を平行移動、回転、青い2つの点を軸にした回転
PCD2：2つ目の点群を平行移動、回転、青い2つの点を軸にした回転
VecStart：青い点の始点を平行移動
VecEnd：青い点の終点を平行移動


平行移動と回転の大きさ（arg）は数字キーで常に変更可能


最後の出力結果はメモしておいてその変換行列を全ての点群にかける
"""

class TranslateMode(Enum):
    PCD1 = 1
    PCD2 = 2
    VecStart = 3
    VecEnd = 4



def create_arbitrary_rotate_matrix(vecStart:np.ndarray, vecEnd:np.ndarray, theta: np.float64) -> np.ndarray:
    """任意軸回転をする回転行列を生成

    Args:
        vecStart (x:float, y:float, z:float): 任意軸ベクトルの始点
        vecEnd (x:float, y:float, z:float): 任意軸ベクトルの終点
        theta (float): 回転角

    Returns:
        np.ndarray: 変換行列 (4x4)
    """
    sin, cos = math.sin(math.radians(theta)), math.cos(math.radians(theta))
    n = vecEnd - vecStart
    n = n / np.linalg.norm(n)
    #print("n", n)
    mat = np.array( [
                    [n[0]*n[0]*(1-cos)+cos,      n[0]*n[1]*(1-cos)+n[2]*sin, n[0]*n[2]*(1-cos)-n[1]+sin, 0.0],
                    [n[0]*n[1]*(1-cos)-n[2]*sin, n[1]*n[1]*(1-cos)+cos,      n[1]*n[2]*(1-cos)+n[0]*sin, 0.0],
                    [n[0]*n[2]*(1-cos)+n[1]*sin, n[1]*n[2]*(1-cos)-n[0]*sin, n[2]*n[2]*(1-cos)+cos,      0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    ], dtype=np.float64)
    return mat
    translator = np.identity(4)
    translator[:3,3] = vecStart.T
    translator[3,3]  = 1.0
    #print(translator)
    mat = np.dot(mat, translator)
    translator[:3,3] = (-vecStart).T
    #print(translator)
    return np.dot(translator, mat)
    
    
#def apply_calibration_arbitrary_axis(points:np.ndarray, dX:np.float64, dY:np.float64, dZ:np.float64, vecStart:np.ndarray, vecEnd:np.ndarray, theta:np.float64) -> np.ndarray:
def apply_calibration_arbitrary_axis(points:np.ndarray, vecStart:np.ndarray, vecEnd:np.ndarray, theta:np.float64) -> np.ndarray:
    """点群を任意軸回転

    Args:
        points (np.ndarray): 変換元の点群. [[x, y, z], ...]
        vecStart (x:float, y:float, z:float): 任意軸ベクトルの始点
        vecEnd (x:float, y:float, z:float): 任意軸ベクトルの終点
        theta (float): 回転角

    Returns:
        np.ndarray: [[x, y, z], ...]
    """
    """
    hom = to_hom(points)
    translator = np.identity(4)
    translator[:,3] = np.array([dX, dY, dZ, 1], dtype=np.float64).T
    #hom = np.dot(hom, translator.T)
    rotator = create_arbitrary_rotate_matrix(vecStart-[dX, dY, dZ], vecEnd-[dX, dY, dZ], theta)
    mat = np.dot(translator, rotator)

    return np.dot(hom, mat.T)[:,:3]
    """
    hom = to_hom(points)
    rotator = create_arbitrary_rotate_matrix(vecStart, vecEnd, theta)

    return np.dot(hom, rotator.T)[:,:3]


def apply_calibration(points: np.ndarray, mat:np.ndarray) -> np.ndarray:
    """点群にキャリブレーションを適用

    Args:
        points (np.ndarray): 変換元の点群. [[x, y, z], ...]
        dX (float): X 平行移動
        dY (float): Y 平行移動
        dZ (float): Z 平行移動
        dAzimuth (float): 方位角
        dVertical (float): 仰俯角

    Returns:
        np.ndarray: [[x, y, z], ...]
    """

    hom = to_hom(points)

    
    return np.dot(hom, mat.T)[:,:3]



def transform_target_interactively(points1:np.ndarray, points2:np.ndarray, scale=1, initial_parameter1=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], initial_parameter2=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) -> np.ndarray:
    """2つの点群を手動で可視化しながら位置合わせ

    """

    pcd1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points1))
    pcd2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points2))
    # Option
    xy_plane_hight = 0.0

    #o3d.visualization.draw_geometries([pcd1, pcd2])

    # variables
    #delta = np.array([0, 0, 0, 0, 0], dtype=np.float)
    parameters1 = np.array(initial_parameter1, dtype=np.float64)
    parameters2 = np.array(initial_parameter2, dtype=np.float64)
    mat1 = np.identity(4)
    mat2 = np.identity(4)

    move_size = np.float64(1.0)
    mode = TranslateMode.PCD2
    pcd1 = np_pcd(points1)
    pcd2 = np_pcd(points2)
    pcd1.paint_uniform_color([0.7, 0.7, 0.7])
    pcd2.paint_uniform_color([1.0, 0.0, 0.0])


    vecStart = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    vecEnd = np.array([0.0, 0.0, 10.0], dtype=np.float64)
    vec = o3d.geometry.PointCloud()
    vec.points = o3d.utility.Vector3dVector(np.stack([vecStart, vecEnd]))
    vec.paint_uniform_color([0.0, 0.0, 1.0])
    theta = 0.0

    #xy_plane = create_xy_plane_LineSet(0.2*scale, 200, xy_plane_hight)
    #xy_plane = create_xy_plane_LineSet(100, 500, -1730)
    xy_plane = create_xy_plane_LineSet(100, 100, 0)
    yz_plane_max = create_yz_plnae_LineSet(1, 1, 69.120)
    yz_plane_min = create_yz_plnae_LineSet(1, 1, 0.0)
    zx_plane_max = create_zx_plnae_LineSet(1, 1, 39.680)
    zx_plane_min = create_zx_plnae_LineSet(1, 1, -39.680)

    # update target point cloud with current parameters
    def update_geometry() -> None:
        nonlocal parameters1, parameters2, mat1, mat2, vecStart, vecEnd, vec, pcd1, pcd2, points1, points2
        
        nmat1 = create_rotate3_matrix_telecognix(parameters1[0], parameters1[1], parameters1[2], parameters1[3], parameters1[4], parameters1[5])
        nmat2 = create_rotate3_matrix_telecognix(parameters2[0], parameters2[1], parameters2[2], parameters2[3], parameters2[4], parameters2[5])
        
        pcd1.points = o3d.utility.Vector3dVector(apply_calibration(apply_calibration(points1, mat1), nmat1))
        pcd2.points = o3d.utility.Vector3dVector(apply_calibration(apply_calibration(points2, mat2), nmat2))
        vec.points = o3d.utility.Vector3dVector(np.stack([vecStart, vecEnd]))

    def update_geometry_arbitrary_rotate() -> None:
        nonlocal mode, theta, vecStart, vecEnd, pcd1, mat1, points1, pcd2, mat2, points2
        if mode == TranslateMode.PCD1:
            pcd1.points = o3d.utility.Vector3dVector(apply_calibration_arbitrary_axis(apply_calibration(points1, mat1) , vecStart, vecEnd, theta))
        elif mode == TranslateMode.PCD2:
            pcd2.points = o3d.utility.Vector3dVector(apply_calibration_arbitrary_axis(apply_calibration(points2, mat2) , vecStart, vecEnd, theta))

    def check_and_save_nomal_translate(mode:TranslateMode) -> None:
        nonlocal parameters1, mat1, parameters2, mat2
        if mode == TranslateMode.PCD1:
            if not np.allclose(parameters1, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]):
                rotator = create_rotate3_matrix_telecognix(parameters1[0], parameters1[1], parameters1[2], parameters1[3], parameters1[4], parameters1[5])
                mat1 = np.dot(rotator, mat1)
                #parameters1 = np.zeros(6, dtype=np.float64)
        if mode == TranslateMode.PCD2:
            if not np.allclose(parameters2, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]):
                rotator = create_rotate3_matrix_telecognix(parameters2[0], parameters2[1], parameters2[2], parameters2[3], parameters2[4], parameters2[5])
                mat2 = np.dot(rotator, mat2)
                print(parameters2)
                print(mat2)
                #parameters2 = np.zeros(6, dtype=np.float64)


    def check_and_save_arbitrary_rotate(mode:TranslateMode) -> None:
        nonlocal theta, vecStart, vecEnd, mat1, mat2

        #if theta != 0.0 :
        #    arbitrary_rotator = create_arbitrary_rotate_matrix(vecStart, vecEnd, theta)
        #    theta = np.float64(0.0)
        #    if mode == TranslateMode.PCD1:
        #        mat1 = np.dot(arbitrary_rotator, mat1)
        #    elif mode == TranslateMode.PCD2:
        #        mat2 = np.dot(arbitrary_rotator, mat2)
        #        print(mat2)
        #    else :
        #        print("check_and_save_arbitrary_rotate : mode =", mode)

    # initialize geometry
    update_geometry()


    # keybind callbacks

    def translate_xaxis_plus(vis):
        nonlocal move_size, parameters1, parameters2, vecStart, vecEnd, mode
        check_and_save_arbitrary_rotate(mode)
        if mode == TranslateMode.PCD1:
            parameters1[0] += move_size
        elif mode == TranslateMode.PCD2:
            parameters2[0] += move_size
        elif mode == TranslateMode.VecStart:
            vecStart[0] += move_size
        elif mode == TranslateMode.VecEnd:
            vecEnd[0] += move_size
        else:
            print("translate_xaxis_plus : unknown TranslateMode")
            return False
        update_geometry()
        return True
    
    def translate_xaxis_minus(vis):
        nonlocal move_size, parameters1, parameters2, vecStart, vecEnd, mode
        check_and_save_arbitrary_rotate(mode)
        if mode == TranslateMode.PCD1:
            parameters1[0] -= move_size
        elif mode == TranslateMode.PCD2:
            parameters2[0] -= move_size
        elif mode == TranslateMode.VecStart:
            vecStart[0] -= move_size
        elif mode == TranslateMode.VecEnd:
            vecEnd[0] -= move_size
        else:
            print("translate_xaxis_minus : unknown TranslateMode")
            return False
        update_geometry()
        return True

    def translate_yaxis_plus(vis):
        nonlocal move_size, parameters1, parameters2, vecStart, vecEnd, mode
        check_and_save_arbitrary_rotate(mode)
        if mode == TranslateMode.PCD1:
            parameters1[1] += move_size
        elif mode == TranslateMode.PCD2:
            parameters2[1] += move_size
        elif mode == TranslateMode.VecStart:
            vecStart[1] += move_size
        elif mode == TranslateMode.VecEnd:
            vecEnd[1] += move_size
        else:
            print("translate_yaxis_plus : unknown TranslateMode")
            return False
        update_geometry()
        return True

    def translate_yaxis_minus(vis):
        nonlocal move_size, parameters1, parameters2, vecStart, vecEnd, mode
        check_and_save_arbitrary_rotate(mode)
        if mode == TranslateMode.PCD1:
            parameters1[1] -= move_size
        elif mode == TranslateMode.PCD2:
            parameters2[1] -= move_size
        elif mode == TranslateMode.VecStart:
            vecStart[1] -= move_size
        elif mode == TranslateMode.VecEnd:
            vecEnd[1] -= move_size
        else:
            print("translate_yaxis_minus : unknown TranslateMode")
            return False
        update_geometry()
        return True

    def translate_zaxis_plus(vis):
        nonlocal move_size, parameters1, parameters2, vecStart, vecEnd, mode
        check_and_save_arbitrary_rotate(mode)
        if mode == TranslateMode.PCD1:
            parameters1[2] += move_size
        elif mode == TranslateMode.PCD2:
            parameters2[2] += move_size
        elif mode == TranslateMode.VecStart:
            vecStart[2] += move_size
        elif mode == TranslateMode.VecEnd:
            vecEnd[2] += move_size
        else:
            print("translate_zaxis_plus : unknown TranslateMode")
            return False
        update_geometry()
        return True

    def translate_zaxis_minus(vis):
        nonlocal move_size, parameters1, parameters2, vecStart, vecEnd, mode
        check_and_save_arbitrary_rotate(mode)
        if mode == TranslateMode.PCD1:
            parameters1[2] -= move_size
        elif mode == TranslateMode.PCD2:
            parameters2[2] -= move_size
        elif mode == TranslateMode.VecStart:
            vecStart[2] -= move_size
        elif mode == TranslateMode.VecEnd:
            vecEnd[2] -= move_size
        else:
            print("translate_zaxis_minus : unknown TranslateMode")
            return False
        update_geometry()
        return True

    def rotate_xaxis_plus(vis):
        nonlocal move_size, parameters1, parameters2, mode, scale
        check_and_save_arbitrary_rotate(mode)        
        if mode == TranslateMode.PCD1:
            parameters1[3] += move_size/scale
        elif mode == TranslateMode.PCD2:
            parameters2[3] += move_size/scale
        else :
            print("cannot rotate azimuth left. mode =", mode)
            return False
        update_geometry()
        return True

    def rotate_xaxis_minus(vis):
        nonlocal move_size, parameters1, parameters2, mode, scale
        check_and_save_arbitrary_rotate(mode)
        if mode == TranslateMode.PCD1:
            parameters1[3] -= move_size/scale
        elif mode == TranslateMode.PCD2:
            parameters2[3] -= move_size/scale
        else:
            print("cannot rotate azimuth right. mode =", mode)
            return False
        update_geometry()
        return True

    def rotate_yaxis_plus(vis):
        nonlocal move_size, parameters1, parameters2, mode, scale
        check_and_save_arbitrary_rotate(mode)
        if mode == TranslateMode.PCD1:
            parameters1[4] += move_size/scale
        elif mode == TranslateMode.PCD2:
            parameters2[4] += move_size/scale
        else:
            print("cannot rotate vertical up. mode =", mode)
            return False
        update_geometry()
        return True

    def rotate_yaxis_minus(vis):
        nonlocal move_size, parameters1, parameters2, mode, scale
        check_and_save_arbitrary_rotate(mode)
        if mode == TranslateMode.PCD1:
            parameters1[4] -= move_size/scale
        elif mode == TranslateMode.PCD2:
            parameters2[4] -= move_size/scale
        else:
            print("cannot rotate vertical down. mode =", mode)
            return False
        update_geometry()
        return True

    def rotate_zaxis_plus(vis):
        nonlocal move_size, parameters1, parameters2, mode, scale
        check_and_save_arbitrary_rotate(mode)
        if mode == TranslateMode.PCD1:
            parameters1[5] += move_size/scale
        elif mode == TranslateMode.PCD2:
            parameters2[5] += move_size/scale
        else:
            print("cannot rotate vertical up. mode =", mode)
            return False
        update_geometry()
        return True

    def rotate_zaxis_minus(vis):
        nonlocal move_size, parameters1, parameters2, mode, scale
        check_and_save_arbitrary_rotate(mode)
        if mode == TranslateMode.PCD1:
            parameters1[5] -= move_size/scale
        elif mode == TranslateMode.PCD2:
            parameters2[5] -= move_size/scale
        else:
            print("cannot rotate vertical down. mode =", mode)
            return False
        update_geometry()
        return True

    def increase_move_size001(vis):
        nonlocal move_size, scale
        move_size += 0.01 * scale
        print("move_size =", move_size/scale)
        return False
        
    def increase_move_size01(vis):
        nonlocal move_size, scale
        move_size += 0.1 * scale
        print("move_size =", move_size/scale)
        return False

    def increase_move_size1(vis):
        nonlocal move_size, scale
        move_size += 1.0 * scale
        print("move_size =", move_size/scale)
        return False

    def increase_move_size10(vis):
        nonlocal move_size, scale
        move_size += 10.0 * scale
        print("move_size =", move_size/scale)
        return False

    def increase_move_size100(vis):
        nonlocal move_size, scale
        move_size += 100.0 * scale
        print("move_size =", move_size/scale)
        return False

    def decrease_move_size001(vis):
        nonlocal move_size, scale
        move_size -= 0.01 * scale
        if move_size < 0.0:
            move_size = np.float64(0.0)
        print("move_size =", move_size/scale)
        return False

    def decrease_move_size01(vis):
        nonlocal move_size, scale
        move_size -= 0.1 * scale
        if move_size < 0.0:
            move_size = np.float64(0.0)
        print("move_size =", move_size/scale)
        return False

    def decrease_move_size1(vis):
        nonlocal move_size, scale
        move_size -= 1.0 * scale
        if move_size < 0.0:
            move_size = np.float64(0.0)
        print("move_size =", move_size/scale)
        return False

    def decrease_move_size10(vis):
        nonlocal move_size, scale
        move_size -= 10.0 * scale
        if move_size < 0.0:
            move_size = np.float64(0.0)
        print("move_size =", move_size/scale)
        return False


    
    def decrease_move_size100(vis):
        nonlocal move_size, scale
        move_size -= 100.0 * scale
        if move_size < 0.0:
            move_size = np.float64(0.0)
        print("move_size =", move_size/scale)
        return False
    
    def print_parameters(vis):
        nonlocal parameters1, parameters2
        print(parameters1)
        print(parameters2)
        return False

    def start_translate_pcd_mode(vis):
        nonlocal mode, pcd1, pcd2
        if mode == TranslateMode.PCD1:
            check_and_save_arbitrary_rotate(mode)
            mode = TranslateMode.PCD2
            pcd2.paint_uniform_color([1.0, 0.0, 0.0])
            pcd1.paint_uniform_color([0.7, 0.7, 0.7])
        elif mode == TranslateMode.PCD2:
            check_and_save_arbitrary_rotate(mode)
            mode= TranslateMode.PCD1
            pcd1.paint_uniform_color([1.0, 0.0, 0.0])
            pcd2.paint_uniform_color([0.7, 0.7, 0.7])
        else :
            mode = TranslateMode.PCD1
            pcd1.paint_uniform_color([1.0, 0.0, 0.0])
            pcd2.paint_uniform_color([0.7, 0.7, 0.7])
        print("mode =", mode)
        return True
    
    def start_translate_vecStart_mode(vis):
        return False
        nonlocal mode
        mode = TranslateMode.VecStart
        pcd1.paint_uniform_color([0.7, 0.7, 0.7])
        pcd2.paint_uniform_color([0.7, 0.7, 0.7])
        print("mode =", mode)
        return True

    def start_translate_vecEnd_mode(vis):
        return False
        nonlocal mode
        mode = TranslateMode.VecEnd
        pcd1.paint_uniform_color([0.7, 0.7, 0.7])
        pcd2.paint_uniform_color([0.7, 0.7, 0.7])
        print("mode =", mode)
        return True

    def arbitrary_rotate_plus(vis):
        nonlocal mode, theta, move_size
        #check_and_save_nomal_translate(mode)
        #theta += move_size
        #update_geometry_arbitrary_rotate()
        #return True

    def arbitrary_rotate_minus(vis):
        nonlocal mode, theta, move_size
        #check_and_save_nomal_translate(mode)
        #theta -= move_size
        #update_geometry_arbitrary_rotate()
        #return True

    is_visible = True
    def change_xy_plane_visibility(vis):
        nonlocal is_visible, xy_plane
        is_visible = not is_visible
        if is_visible:
            vis.add_geometry(xy_plane, False)
            vis.add_geometry(yz_plane_min, False)
            vis.add_geometry(yz_plane_max, False)
            vis.add_geometry(zx_plane_min, False)
            vis.add_geometry(zx_plane_max, False)
        else :
            vis.remove_geometry(xy_plane, False)
            vis.remove_geometry(yz_plane_min, False)
            vis.remove_geometry(yz_plane_max, False)
            vis.remove_geometry(zx_plane_min, False)
            vis.remove_geometry(zx_plane_max, False)
        return True


    callback_list ={ord("U"):translate_xaxis_plus, ord("J"):translate_xaxis_minus,
                    ord("I"):translate_yaxis_plus, ord("K"):translate_yaxis_minus,
                    ord("O"):translate_zaxis_plus, ord("L"):translate_zaxis_minus,
                    ord("R"):rotate_xaxis_plus, ord("F"):rotate_xaxis_minus,
                    ord("D"):rotate_zaxis_plus, ord("G"):rotate_zaxis_minus,
                    ord("1"):increase_move_size001, ord("2"):increase_move_size01,
                    ord("3"):increase_move_size1, ord("4"):increase_move_size10,
                    ord("5"):increase_move_size100, ord("0"):decrease_move_size001,
                    ord("9"):decrease_move_size01, ord("8"):decrease_move_size1,
                    ord("7"):decrease_move_size10, ord("6"):decrease_move_size100,
                    ord("C"):start_translate_pcd_mode, ord("V"):start_translate_vecStart_mode,
                    ord("B"):start_translate_vecEnd_mode, ord("Z"):rotate_yaxis_minus,
                    ord("X"):rotate_yaxis_plus, ord("M"):change_xy_plane_visibility,
                    ord("P"):print_parameters}

    usage_message = "Usage\n" + \
                    "-- Mouse view control --\n" + \
                    "Left button + drag         : Rotate\n" + \
                    "Ctrl + left button + drag  : Translate\n" + \
                    "Wheel button + drag        : Translate\n" + \
                    "Shift + left button + drag : Roll\n" + \
                    "Wheel                      : Zoom in/out\n" + \
                    "-- Keyboard control --\n" + \
                    "U : Move to +x\n" + \
                    "J : Move to -x\n" + \
                    "I : Move to +y\n" + \
                    "K : Move to -y\n" + \
                    "O : Move to +z\n" + \
                    "L : Move to -z\n" + \
                    "R : Rotate xaxis plus \n" + \
                    "F : Rotate xaxis minus \n" + \
                    "D : Rotate zaxis plus\n" + \
                    "G : Rotate zaxis minus\n" + \
                    "X : Rotate yaxis plus\n" + \
                    "Z : Rotate yaxis minus\n" + \
                    "1 : Increase move_size by 0.01\n" + \
                    "2 : Increase move_size by 0.1\n" + \
                    "3 : Increase move_size by 1\n" + \
                    "4 : Increase move_size by 10\n" + \
                    "5 : Increase move_size by 100\n" + \
                    "0 : Decrease move_size by 0.01\n" + \
                    "9 : Decrease move_size by 0.1\n" + \
                    "8 : Decrease move_size by 1\n" + \
                    "7 : Decrease move_size by 10\n" + \
                    "6 : Decrease move_size by 100\n" + \
                    "C : Change mode to translate pcd\n" + \
                    "V : Change mode to move vecStart\n" + \
                    "B : Change mode to move vecEnd\n" + \
                    "M : make xy plane visible/invisible\n" + \
                    "P: Print label\n" + \
                    "Q : Quit"
    print(usage_message)

    #o3d.visualization.draw_geometries_with_key_callbacks([pcd1, pcd2, vec, xy_plane], key_to_callback=callback_list)
    o3d.visualization.draw_geometries_with_key_callbacks([pcd1, pcd2, vec, xy_plane, yz_plane_min, yz_plane_max, zx_plane_min, zx_plane_max], key_to_callback=callback_list)

    check_and_save_arbitrary_rotate(mode)
    check_and_save_nomal_translate(TranslateMode.PCD1)
    check_and_save_nomal_translate(TranslateMode.PCD2)
    return parameters1, mat1, parameters2, mat2

def main() :
    
    # Option
    #127.0.0.1-9010_1_1662439120642232のような形式ならLiDAR
    #000001.binのような形式ならKITTI
    file_type = "LiDAR"

    # Option
    #1フレームより点を濃くするために複数のフレームを重ねてキャリブレーションに使用
    frame_stack_num = 1

    # 点群読み込み
    dirpath_base = sys.argv[1]
    files_base = get_files(dirpath_base, "pcd", file_type)
    start_frame_base = int(sys.argv[2])
    points_base = np.vstack([open_bin(dirpath_base + "/" + files_base[i])[:, :3] for i in range(start_frame_base, start_frame_base+frame_stack_num)])


    dirpath_target = sys.argv[3]
    files_target = get_files(dirpath_target, "pcd", file_type)
    start_frame_target = int(sys.argv[4])
    points_target = np.vstack([open_bin(dirpath_target + "/" + files_target[i])[:, :3] for i in range(start_frame_target, start_frame_target+frame_stack_num)])

    # スケールの設定
    scale = 1
    if len(sys.argv) >= 6:
        scale = int(sys.argv[5])

    # Option
    # 外れ値の除去
    # points_target = points_target[np.all(np.abs(points_target) < 1000.0*scale, axis=1), :]
    # points_base = points_base[np.all(np.abs(points_base) < 1000.0*scale, axis=1), :]
    #points_target = points_target[points_target[:, 2] < 5000]
    #points_base = points_base[points_base[:, 2] < 5000]

    # points_base = points_base[points_base[:, 0] > -20000]
    # points_base = points_base[points_base[:, 0] < 20000]
    # points_target = points_target[points_target[:, 0] > -20000]
    # points_target = points_target[points_target[:, 0] < 20000]
    # points_base = points_base[points_base[:, 1] > -20000]
    # points_base = points_base[points_base[:, 1] < 20000]
    # points_target = points_target[points_target[:, 1] > -20000]
    # points_target = points_target[points_target[:, 1] < 20000]
    print("points base shape =", points_base.shape)
    print("points target shape =", points_target.shape)
    
    
    #parameter_base, mat_base, parameter_target, mat_target = transform_target_interactively(points_base, points_target, scale=scale, initial_parameter2=[ 1.860e+04,  1.281e+04,  1.390e+03, -2.300e+00, -3.000e-01, -1.227e+02])
    parameter_base, mat_base, parameter_target, mat_target = transform_target_interactively(points_base, points_target, scale=scale)
    
    roll_base = parameter_base[3]
    parameter_base[3] = parameter_base[5]
    parameter_base[5] = roll_base

    roll_target = parameter_target[3]
    parameter_target[3] = parameter_target[5]
    parameter_target[5] = roll_target

    print(dirpath_base + ":")
    print(parameter_base)
    # print(mat_base)
    print("[")
    for row in mat_base:
    # 各行を [要素,要素,要素,要素], の形式で出力
        print(" [" + ",".join(str(x) for x in row) + "],")
    print("]")

    print(dirpath_target + ":")
    print(parameter_target)
    # print(mat_target)
    print("[")
    for row in mat_target:
        # 各行を [要素,要素,要素,要素], の形式で出力
        print(" [" + ",".join(str(x) for x in row) + "],")
    print("]")



if __name__ == "__main__" :
    main()
