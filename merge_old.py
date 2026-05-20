import sys
from file_util import get_files, open_bin, write_bin
import numpy as np
from pcd_util import apply_calibration

"""
1つの点群データを回転・移動させてもう一つの点群データとマージした点群を保存する

回転行列Rを書き換える必要あり

使い方
python3 adjust_calibration_params.py [点群ディレクトリパス1] [点群ディレクトリパス2] [保存先ディレクトリ]

"""

def main():
  source1_dirpass = sys.argv[1]
  source2_dirpass = sys.argv[2]
  dist_dir = sys.argv[3]
       
  # 書き換え
  R1 = [[-9.99865423e-01, -1.58094685e-02, -4.38138229e-03,  8.31500000e+01],      
            [ 1.64053590e-02, -9.63547392e-01, -2.67034245e-01,  1.99841000e+03],      
            [             0.,     -0.26488299,      0.96428056,           181.5],
            [             0.,              0.,              0.,              1.]]
  R1 = np.array(R1, dtype=np.float64) 

  R2 = [[-9.99865423e-01, -1.58094685e-02, -4.38138229e-03,  8.31500000e+01],
            [ 1.64053590e-02, -9.63547392e-01, -2.67034245e-01,  1.99841000e+03],
            [ 0.00000000e+00, -2.67070187e-01,  9.63677080e-01,  1.82900000e+02],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]
  R2 = np.array(R2, dtype=np.float64) 

  s1_bin_files = get_files(source1_dirpass, "pcd", mode="LiDAR")
  s2_bin_files = get_files(source2_dirpass, "pcd", mode="LiDAR")

  for s1_file_name, s2_file_name in zip(s1_bin_files, s2_bin_files):
      s1_points = open_bin(source1_dirpass  + "/" + s1_file_name)[:,:3]
      s2_points = open_bin(source2_dirpass  + "/" + s2_file_name)[:,:3]
      s1_points = apply_calibration(s1_points, R1)
      s2_points = apply_calibration(s2_points, R2)
      write_bin(dist_dir + s1_file_name, np.concatenate([s1_points, s2_points]), False)
    
if __name__ == "__main__":
    main()