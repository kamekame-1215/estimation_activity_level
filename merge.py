import sys
import os
import numpy as np
from file_util import get_files, open_bin, write_bin
from pcd_util import apply_calibration
from multiprocessing import Pool
from tqdm import tqdm

"""
1つの点群データを回転・移動させてもう一つの点群データとマージした点群を保存する

回転行列Rを書き換える必要あり

使い方
python3 merge.py [点群ディレクトリパス1] [点群ディレクトリパス2] [保存先ディレクトリ]
"""

def process_files(args):
      s1_file_name, s2_file_name, source1_dirpass, source2_dirpass, dist_dir, R1, R2 = args
      # 点群の読み込み
      s1_points = open_bin(os.path.join(source1_dirpass, s1_file_name))[:, :3]
      s2_points = open_bin(os.path.join(source2_dirpass, s2_file_name))[:, :3]

      # 校正適用(行列演算は十分高速だが、ここもNumPyを使って最大限ベクトル化)
      s1_points = apply_calibration(s1_points, R1)
      s2_points = apply_calibration(s2_points, R2)

      # マージした点群データを保存
      output_path = os.path.join(dist_dir, s1_file_name)
      write_bin(output_path, np.concatenate([s1_points, s2_points]), False)

def main():
      source1_dirpass = sys.argv[1]
      source2_dirpass = sys.argv[2]
      dist_dir = sys.argv[3]
      os.makedirs(dist_dir, exist_ok=True)

      # 書き換えが必要な回転行列
      R1 = [
            [0.9997426093226983,0.02183016058893129,0.006177312789888828,-64.20000000000003],
            [-0.02268733357278138,0.961970327588219,0.2722101646433726,-8.000000000000002],
            [0.0,-0.27228024704057424,0.9622179935292854,197.10000000000002],
            [0.0,0.0,0.0,1.0],
            ]
      R1 = np.array(R1, dtype=np.float64) 

      R2 = [
            [-0.9996101150403544,0.026970231955407178,0.007226651872131412,-60.600000000000016],
            [-0.02792163872356791,-0.9655492263372649,-0.2587181354495654,2007.4999999999932],
            [0.0,-0.25881904510252074,0.9659258262890683,175.1999999999998],
            [0.0,0.0,0.0,1.0],
            ]
      R2 = np.array(R2, dtype=np.float64) 

      # 入力ファイルリスト取得
      s1_bin_files = get_files(source1_dirpass, "pcd", mode="LiDAR")
      s2_bin_files = get_files(source2_dirpass, "pcd", mode="LiDAR")

      # ペアになっているファイルを並列処理用にパッキング
      args_list = [(s1_file_name, s2_file_name, source1_dirpass, source2_dirpass, dist_dir, R1, R2) 
                  for s1_file_name, s2_file_name in zip(s1_bin_files, s2_bin_files)]

      # 並列化
      with Pool() as pool:
            for _ in tqdm(pool.imap(process_files, args_list), total=len(args_list), desc="Merging"):
                  pass

if __name__ == "__main__":
      main()
