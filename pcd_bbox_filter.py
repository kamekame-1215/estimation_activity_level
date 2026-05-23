#!/usr/bin/env python3
"""
PCD BBox Filter

Open3D上でバウンディングボックス(BBox)を対話的に指定し、
エリア外の点群を除去して新しいフォルダに保存するツール。

使い方:
    python pcd_bbox_filter.py <入力PCD or ディレクトリ> <出力ディレクトリ>
    python pcd_bbox_filter.py <入力PCD or ディレクトリ> <出力ディレクトリ> --bbox xmin,xmax,ymin,ymax,zmin,zmax

ビジュアライザのキー操作:
    6: BBox座標をコンソールから入力
       入力形式: xmin,xmax,ymin,ymax,zmin,zmax
       r: BBoxをリセット / q: キャンセル
    s: フィルタリングを確定して保存・終了
    q: 保存せずに終了

点群の色:
    緑色: BBox内 (保持される)
    灰色: BBox外 (除去される)
    赤枠: BBox表示
"""

import os
import sys
import glob
import argparse
from typing import List, Optional, Tuple

import numpy as np
import open3d as o3d


def create_bbox_lineset(region: Tuple[float, ...]) -> o3d.geometry.LineSet:
    xmin, xmax, ymin, ymax, zmin, zmax = region
    corners = np.array([
        [xmin, ymin, zmin], [xmin, ymin, zmax],
        [xmin, ymax, zmin], [xmin, ymax, zmax],
        [xmax, ymin, zmin], [xmax, ymin, zmax],
        [xmax, ymax, zmin], [xmax, ymax, zmax],
    ], dtype=np.float64)
    edges = np.array([
        [0, 1], [0, 2], [0, 4],
        [1, 3], [1, 5],
        [2, 3], [2, 6],
        [3, 7],
        [4, 5], [4, 6],
        [5, 7],
        [6, 7],
    ], dtype=np.int32)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners)
    ls.lines = o3d.utility.Vector2iVector(edges)
    ls.colors = o3d.utility.Vector3dVector(np.tile([1.0, 0.0, 0.0], (len(edges), 1)))
    return ls


def build_inside_mask(points: np.ndarray, region: Tuple[float, ...]) -> np.ndarray:
    xmin, xmax, ymin, ymax, zmin, zmax = region
    return (
        (points[:, 0] >= xmin) & (points[:, 0] <= xmax) &
        (points[:, 1] >= ymin) & (points[:, 1] <= ymax) &
        (points[:, 2] >= zmin) & (points[:, 2] <= zmax)
    )


def filter_pcd_by_bbox(
    pcd: o3d.geometry.PointCloud, region: Tuple[float, ...]
) -> o3d.geometry.PointCloud:
    points = np.asarray(pcd.points)
    mask = build_inside_mask(points, region)
    return pcd.select_by_index(np.where(mask)[0])


def get_pcd_files(input_path: str) -> List[str]:
    if os.path.isfile(input_path):
        return [input_path]
    if os.path.isdir(input_path):
        files = sorted(glob.glob(os.path.join(input_path, "*.pcd")))
        if not files:
            print(f"警告: {input_path} にPCDファイルが見つかりませんでした。")
        return files
    raise FileNotFoundError(f"入力パスが見つかりません: {input_path}")


def print_pcd_stats(points: np.ndarray, label: str = "") -> None:
    if label:
        print(f"[{label}]")
    print(f"  点数: {len(points)}")
    if len(points) > 0:
        print(f"  X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
        print(f"  Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
        print(f"  Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")


def interactive_bbox_selection(
    ref_pcd: o3d.geometry.PointCloud,
) -> Optional[Tuple[float, ...]]:
    """
    Open3D ビジュアライザでインタラクティブにBBoxを指定する。
    確定した region (xmin,xmax,ymin,ymax,zmin,zmax) またはNone(キャンセル時)を返す。
    """
    original_points = np.asarray(ref_pcd.points).copy()

    region_holder: List[Optional[Tuple[float, ...]]] = [None]
    bbox_ls_holder: List[Optional[o3d.geometry.LineSet]] = [None]
    confirmed = [False]

    # 表示用点群(全点を保持し色だけ変更して更新する)
    display_pcd = o3d.geometry.PointCloud()
    display_pcd.points = o3d.utility.Vector3dVector(original_points)
    display_pcd.paint_uniform_color([0.5, 0.5, 0.5])

    def update_point_colors(vis: o3d.visualization.Visualizer) -> None:
        region = region_holder[0]
        if region is None:
            colors = np.tile([0.5, 0.5, 0.5], (len(original_points), 1))
        else:
            mask = build_inside_mask(original_points, region)
            colors = np.where(
                mask[:, np.newaxis],
                np.array([0.0, 0.8, 0.0]),   # 緑: BBox内
                np.array([0.5, 0.5, 0.5]),   # 灰: BBox外
            )
        display_pcd.colors = o3d.utility.Vector3dVector(colors)
        vis.update_geometry(display_pcd)

    def update_bbox_display(vis: o3d.visualization.Visualizer) -> None:
        if bbox_ls_holder[0] is not None:
            vis.remove_geometry(bbox_ls_holder[0], reset_bounding_box=False)
            bbox_ls_holder[0] = None
        if region_holder[0] is not None:
            ls = create_bbox_lineset(region_holder[0])
            vis.add_geometry(ls, reset_bounding_box=False)
            bbox_ls_holder[0] = ls

    def key_set_bbox(vis: o3d.visualization.Visualizer) -> bool:
        print("\n--- BBox設定 ---")
        print("全点群の座標範囲:")
        print_pcd_stats(original_points)
        print(f"現在のBBox: {region_holder[0]}")
        print("入力形式: xmin,xmax,ymin,ymax,zmin,zmax")
        print("  例: -5.0,5.0,-3.0,3.0,0.0,2.5")
        print("  r: リセット  q: キャンセル")

        line = input("BBox座標 > ").strip()
        if line.lower() == "q":
            print("キャンセルしました。")
            return True
        if line.lower() == "r":
            region_holder[0] = None
            print("BBoxをリセットしました。")
            update_point_colors(vis)
            update_bbox_display(vis)
            vis.update_renderer()
            return True

        try:
            vals = list(map(float, line.split(",")))
            if len(vals) != 6:
                print("エラー: 6つの値をカンマ区切りで入力してください。")
                return True
            region_holder[0] = tuple(vals)  # type: ignore[assignment]
            region = region_holder[0]
            mask = build_inside_mask(original_points, region)
            n_inside = int(mask.sum())
            n_total = len(original_points)
            rate = 100.0 * n_inside / n_total if n_total > 0 else 0.0
            print(f"BBox設定完了: {region}")
            print(f"BBox内の点数: {n_inside} / {n_total} ({rate:.1f}%)")
        except ValueError:
            print("エラー: 数値に変換できませんでした。")
            return True

        update_point_colors(vis)
        update_bbox_display(vis)
        vis.update_renderer()
        return True

    def key_confirm(vis: o3d.visualization.Visualizer) -> bool:
        if region_holder[0] is None:
            print("BBoxが設定されていません。[6]キーで設定してください。")
            return True
        confirmed[0] = True
        vis.close()
        return True

    def key_quit(vis: o3d.visualization.Visualizer) -> bool:
        confirmed[0] = False
        vis.close()
        return True

    print("\n=== PCD BBox Filter ビジュアライザ ===")
    print("  [6]: BBox領域をコンソールから設定")
    print("  [s]: フィルタ確定・保存して終了")
    print("  [q]: 保存せずに終了")
    print("  緑色の点: BBox内 (保持)  灰色の点: BBox外 (除去)")
    print("=======================================\n")

    o3d.visualization.draw_geometries_with_key_callbacks(
        [display_pcd],
        key_to_callback={
            ord("6"): key_set_bbox,
            ord("s"): key_confirm,
            ord("q"): key_quit,
        },
        window_name="PCD BBox Filter  [6]:BBox設定  [s]:保存  [q]:終了",
        width=1280,
        height=720,
    )

    return region_holder[0] if confirmed[0] else None


def process_files(
    pcd_files: List[str], output_dir: str, region: Tuple[float, ...]
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    xmin, xmax, ymin, ymax, zmin, zmax = region
    print(f"\n処理開始: {len(pcd_files)} ファイル")
    print(
        f"BBox: X[{xmin}, {xmax}]  Y[{ymin}, {ymax}]  Z[{zmin}, {zmax}]"
    )
    print(f"出力先: {output_dir}\n")

    for file_path in pcd_files:
        pcd = o3d.io.read_point_cloud(file_path)
        original_count = len(pcd.points)
        filtered_pcd = filter_pcd_by_bbox(pcd, region)
        filtered_count = len(filtered_pcd.points)

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = os.path.join(output_dir, base_name + "_filtered.pcd")
        o3d.io.write_point_cloud(output_path, filtered_pcd)

        rate = 100.0 * filtered_count / original_count if original_count > 0 else 0.0
        print(
            f"  {os.path.basename(file_path)}: "
            f"{original_count} → {filtered_count} 点 ({rate:.1f}%) "
            f"→ {os.path.basename(output_path)}"
        )

    print(f"\n処理完了。フィルタ済みファイルを {output_dir} に保存しました。")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PCD BBox Filter: Open3D上でBBoxを指定してエリア外の点群を除去する",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # インタラクティブ (1ファイル)
  python pcd_bbox_filter.py input.pcd output_dir/

  # インタラクティブ (ディレクトリ内の全PCDファイルを処理)
  python pcd_bbox_filter.py input_dir/ output_dir/

  # BBox直接指定 (GUIなしバッチ処理)
  python pcd_bbox_filter.py input_dir/ output_dir/ --bbox -5,5,-3,3,0,2.5
        """,
    )
    parser.add_argument(
        "input",
        help="入力PCDファイル or PCDファイルが入ったディレクトリ",
    )
    parser.add_argument(
        "output_dir",
        help="フィルタリング後のPCDを保存するディレクトリ",
    )
    parser.add_argument(
        "--bbox", "-b",
        metavar="xmin,xmax,ymin,ymax,zmin,zmax",
        help="BBox座標を直接指定 (省略時はインタラクティブに指定)",
        default=None,
    )
    args = parser.parse_args()

    # 入力ファイル取得
    try:
        pcd_files = get_pcd_files(args.input)
    except FileNotFoundError as e:
        print(f"エラー: {e}")
        sys.exit(1)

    if not pcd_files:
        sys.exit(1)

    print(f"対象ファイル数: {len(pcd_files)}")
    for f in pcd_files[:5]:
        print(f"  {f}")
    if len(pcd_files) > 5:
        print(f"  ... 他 {len(pcd_files) - 5} ファイル")

    # BBox決定
    if args.bbox:
        try:
            vals = list(map(float, args.bbox.split(",")))
            if len(vals) != 6:
                print("エラー: --bbox は xmin,xmax,ymin,ymax,zmin,zmax の形式で指定してください。")
                sys.exit(1)
            region: Tuple[float, ...] = tuple(vals)  # type: ignore[assignment]
            print(f"BBox指定 (コマンドライン): {region}")
        except ValueError:
            print("エラー: --bbox の値を数値に変換できませんでした。")
            sys.exit(1)
    else:
        # インタラクティブモード: 最初のファイルで可視化してBBoxを指定
        ref_file = pcd_files[0]
        print(f"\n参照ファイル: {ref_file}")
        ref_pcd = o3d.io.read_point_cloud(ref_file)
        pts = np.asarray(ref_pcd.points)
        print_pcd_stats(pts, os.path.basename(ref_file))

        region = interactive_bbox_selection(ref_pcd)  # type: ignore[assignment]
        if region is None:
            print("BBoxが確定されませんでした。終了します。")
            sys.exit(0)
        print(f"\n確定BBox: {region}")

    # フィルタリングと保存
    process_files(pcd_files, args.output_dir, region)


if __name__ == "__main__":
    main()
