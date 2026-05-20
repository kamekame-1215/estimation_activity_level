from typing import List, Tuple
from file_util import write_bin, get_files
import numpy as np
import sys
import os
from multiprocessing import Pool
from tqdm import tqdm

def open_bin(file_path: str) -> np.ndarray:
    with open(file_path, 'rb') as f:
        point_cloud: np.ndarray = np.fromfile(f, dtype=np.float32)
    point_cloud = point_cloud.reshape(-1, 4)
    return point_cloud

def filter_area(point_cloud: np.ndarray) -> np.ndarray:
    x_coords: np.ndarray = point_cloud[:, 0]
    y_coords: np.ndarray = point_cloud[:, 1]
    z_coords: np.ndarray = point_cloud[:, 2]

    condition: np.ndarray = (
        # (x_coords) &
        (y_coords < 1600)
        # (z_coords)
    )
    filtered_points: np.ndarray = point_cloud[condition]
    return filtered_points

def process_file(args: Tuple[str, str, str]) -> None:
    file_name, input_dir, output_dir = args

    input_file_path: str = os.path.join(input_dir, file_name)

    point_cloud: np.ndarray = open_bin(input_file_path)
    filtered_point_cloud: np.ndarray = filter_area(point_cloud)

    base_name: str = os.path.splitext(os.path.basename(file_name))[0]
    output_file_path: str = os.path.join(output_dir, base_name + ".bin")
    write_bin(output_file_path, filtered_point_cloud, True)

def main() -> None:
    input_dir: str = sys.argv[1]
    output_dir: str = sys.argv[2]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    bin_files: List[str] = get_files(input_dir, "pcd", mode="LiDAR")
    args_list: List[Tuple[str, str, str]] = [(file_name, input_dir, output_dir) for file_name in bin_files]

    with Pool() as pool:
        # imapを使用し、tqdmで進捗を表示
        list(tqdm(pool.imap(process_file, args_list), total=len(args_list), desc="Processing files"))

if __name__ == '__main__':
    main()
