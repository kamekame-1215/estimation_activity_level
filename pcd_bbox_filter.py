#!/usr/bin/env python3
"""
PCD BBox Filter

Open3D上でバウンディングボックス(BBox)を対話的に指定し、
エリア外の点群を除去して新しいフォルダに保存するツール。

使い方:
    python pcd_bbox_filter.py <入力PCD or ディレクトリ> <出力ディレクトリ>
    python pcd_bbox_filter.py <入力PCD or ディレクトリ> <出力ディレクトリ> --bbox xmin,xmax,ymin,ymax,zmin,zmax
    python pcd_bbox_filter.py <入力PCD or ディレクトリ> <出力ディレクトリ> --config bbox.json

ビジュアライザのキー操作:
    6: BBox全パラメータを一括入力 (xmin,xmax,ymin,ymax,zmin,zmax)
    7: 個別パラメータを対話的に微調整
       コマンド例:
         xmin +1.0   → xmin を +1.0 する
         xmax =10.0  → xmax を 10.0 に設定
         ymin -0.5   → ymin を -0.5 する
         step 0.5    → デフォルトステップ幅を 0.5 に変更
         q           → 調整モードを終了
    l: 設定ファイル(JSON)からBBoxを読み込む
    s: フィルタリングを確定して保存・終了 (BBoxをJSONに自動保存)
    q: 保存せずに終了

点群の色:
    緑色: BBox内 (保持される)
    灰色: BBox外 (除去される)
    赤枠: BBox表示
"""

import json
import os
import sys
import glob
import argparse
from typing import List, Optional, Tuple

import numpy as np
import open3d as o3d

# BBoxパラメータ名(インデックス順)
PARAM_NAMES = ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"]
DEFAULT_CONFIG_NAME = "bbox_config.json"


# ---------------------------------------------------------------------------
# BBox / 点群ユーティリティ
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 設定ファイル (JSON) 入出力
# ---------------------------------------------------------------------------

def save_bbox_config(region: Tuple[float, ...], config_path: str) -> None:
    data = {name: val for name, val in zip(PARAM_NAMES, region)}
    with open(config_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"BBox設定を保存しました: {config_path}")


def load_bbox_config(config_path: str) -> Optional[Tuple[float, ...]]:
    if not os.path.exists(config_path):
        print(f"設定ファイルが見つかりません: {config_path}")
        return None
    with open(config_path, "r") as f:
        data = json.load(f)
    try:
        region = tuple(float(data[name]) for name in PARAM_NAMES)
        return region  # type: ignore[return-value]
    except (KeyError, ValueError) as e:
        print(f"設定ファイルの読み込みに失敗しました: {e}")
        return None


# ---------------------------------------------------------------------------
# ファイルユーティリティ
# ---------------------------------------------------------------------------

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


def print_region(region: Tuple[float, ...]) -> None:
    vals = list(region)
    for name, val in zip(PARAM_NAMES, vals):
        print(f"    {name}: {val:.4f}")


# ---------------------------------------------------------------------------
# インタラクティブBBox選択
# ---------------------------------------------------------------------------

def interactive_bbox_selection(
    ref_pcd: o3d.geometry.PointCloud,
    config_path: str,
    initial_region: Optional[Tuple[float, ...]] = None,
) -> Optional[Tuple[float, ...]]:
    """
    Open3D ビジュアライザでインタラクティブにBBoxを指定する。
    確定した region (xmin,xmax,ymin,ymax,zmin,zmax) またはNone(キャンセル時)を返す。
    """
    original_points = np.asarray(ref_pcd.points).copy()

    region_holder: List[Optional[Tuple[float, ...]]] = [initial_region]
    bbox_ls_holder: List[Optional[o3d.geometry.LineSet]] = [None]
    step_holder: List[float] = [1.0]
    confirmed = [False]

    # 表示用点群 (全点保持・色だけ変更)
    display_pcd = o3d.geometry.PointCloud()
    display_pcd.points = o3d.utility.Vector3dVector(original_points)
    display_pcd.paint_uniform_color([0.5, 0.5, 0.5])

    # ------------------------------------------------------------------
    # 内部更新ヘルパー
    # ------------------------------------------------------------------

    def _refresh_colors(vis: o3d.visualization.Visualizer) -> None:
        region = region_holder[0]
        if region is None:
            colors = np.tile([0.5, 0.5, 0.5], (len(original_points), 1))
        else:
            mask = build_inside_mask(original_points, region)
            colors = np.where(
                mask[:, np.newaxis],
                np.array([0.0, 0.8, 0.0]),  # 緑: BBox内
                np.array([0.5, 0.5, 0.5]),  # 灰: BBox外
            )
        display_pcd.colors = o3d.utility.Vector3dVector(colors)
        vis.update_geometry(display_pcd)

    def _refresh_bbox(vis: o3d.visualization.Visualizer) -> None:
        if bbox_ls_holder[0] is not None:
            vis.remove_geometry(bbox_ls_holder[0], reset_bounding_box=False)
            bbox_ls_holder[0] = None
        if region_holder[0] is not None:
            ls = create_bbox_lineset(region_holder[0])
            vis.add_geometry(ls, reset_bounding_box=False)
            bbox_ls_holder[0] = ls

    def _apply_region(vis: o3d.visualization.Visualizer) -> None:
        """region_holder[0] を表示に反映する共通処理"""
        region = region_holder[0]
        if region is not None:
            mask = build_inside_mask(original_points, region)
            n_inside = int(mask.sum())
            n_total = len(original_points)
            rate = 100.0 * n_inside / n_total if n_total > 0 else 0.0
            print(f"  BBox内の点数: {n_inside} / {n_total} ({rate:.1f}%)")
        _refresh_colors(vis)
        _refresh_bbox(vis)
        vis.update_renderer()

    # ------------------------------------------------------------------
    # キーコールバック: 6 → 一括入力
    # ------------------------------------------------------------------

    def key_set_bbox(vis: o3d.visualization.Visualizer) -> bool:
        print("\n--- [6] BBox一括設定 ---")
        print("全点群の座標範囲:")
        print_pcd_stats(original_points)
        if region_holder[0] is not None:
            print("現在のBBox:")
            print_region(region_holder[0])
        print("入力形式: xmin,xmax,ymin,ymax,zmin,zmax  (例: -5.0,5.0,-3.0,3.0,0.0,2.5)")
        print("  r: リセット  q: キャンセル")

        line = input("BBox座標 > ").strip()
        if line.lower() == "q":
            print("キャンセルしました。")
            return True
        if line.lower() == "r":
            region_holder[0] = None
            print("BBoxをリセットしました。")
            _apply_region(vis)
            return True

        try:
            vals = list(map(float, line.split(",")))
            if len(vals) != 6:
                print("エラー: 6つの値をカンマ区切りで入力してください。")
                return True
            region_holder[0] = tuple(vals)  # type: ignore[assignment]
            print("BBoxを設定しました:")
            print_region(region_holder[0])
        except ValueError:
            print("エラー: 数値に変換できませんでした。")
            return True

        _apply_region(vis)
        return True

    # ------------------------------------------------------------------
    # キーコールバック: 7 → 個別パラメータ微調整ループ
    # ------------------------------------------------------------------

    def key_adjust(vis: o3d.visualization.Visualizer) -> bool:
        if region_holder[0] is None:
            print("まず [6] キーでBBoxを設定してください。")
            return True

        print("\n--- [7] 個別パラメータ調整モード ---")
        print(f"デフォルトステップ幅: {step_holder[0]}")
        print("コマンド例:")
        print("  xmin +1.0   → xmin を +1.0 する")
        print("  xmax =10.0  → xmax を 10.0 に設定")
        print("  ymin -0.5   → ymin を -0.5 する")
        print("  step 0.5    → ステップ幅を 0.5 に変更")
        print("  q           → 調整モードを終了")
        print(f"パラメータ: {', '.join(PARAM_NAMES)}")

        vals = list(region_holder[0])

        while True:
            print("\n現在のBBox:")
            for name, v in zip(PARAM_NAMES, vals):
                print(f"  {name}: {v:.4f}")

            line = input("コマンド > ").strip()
            if line.lower() == "q":
                break

            parts = line.split()
            if len(parts) != 2:
                print("エラー: 'パラメータ名 値' の形式で入力してください。")
                continue

            target, expr = parts[0].lower(), parts[1]

            # step 変更
            if target == "step":
                try:
                    step_holder[0] = float(expr)
                    print(f"ステップ幅を {step_holder[0]} に変更しました。")
                except ValueError:
                    print("エラー: ステップ幅は数値で指定してください。")
                continue

            # パラメータ番号 or 名前を解決
            if target in PARAM_NAMES:
                idx = PARAM_NAMES.index(target)
            else:
                print(f"エラー: パラメータ名が不正です。{PARAM_NAMES} のいずれかを指定してください。")
                continue

            # 値を解析 (+1.0 / -1.0 / =5.0)
            try:
                if expr.startswith("="):
                    vals[idx] = float(expr[1:])
                elif expr.startswith("+"):
                    vals[idx] += float(expr[1:]) if len(expr) > 1 else step_holder[0]
                elif expr.startswith("-"):
                    vals[idx] -= float(expr[1:]) if len(expr) > 1 else step_holder[0]
                else:
                    vals[idx] = float(expr)
            except ValueError:
                print("エラー: 値の形式が不正です。例: +1.0 / -0.5 / =3.0")
                continue

            region_holder[0] = tuple(vals)  # type: ignore[assignment]
            _apply_region(vis)

        print("調整モードを終了しました。")
        return True

    # ------------------------------------------------------------------
    # キーコールバック: l → 設定ファイルから読み込み
    # ------------------------------------------------------------------

    def key_load(vis: o3d.visualization.Visualizer) -> bool:
        print(f"\n--- [l] 設定ファイル読み込み: {config_path} ---")
        loaded = load_bbox_config(config_path)
        if loaded is None:
            return True
        region_holder[0] = loaded
        print("読み込んだBBox:")
        print_region(region_holder[0])
        _apply_region(vis)
        return True

    # ------------------------------------------------------------------
    # キーコールバック: s → 確定・保存
    # ------------------------------------------------------------------

    def key_confirm(vis: o3d.visualization.Visualizer) -> bool:
        if region_holder[0] is None:
            print("BBoxが設定されていません。[6]か[7]キーで設定してください。")
            return True
        confirmed[0] = True
        vis.close()
        return True

    # ------------------------------------------------------------------
    # キーコールバック: q → 終了
    # ------------------------------------------------------------------

    def key_quit(vis: o3d.visualization.Visualizer) -> bool:
        confirmed[0] = False
        vis.close()
        return True

    # ------------------------------------------------------------------
    # ビジュアライザ起動
    # ------------------------------------------------------------------

    print("\n=== PCD BBox Filter ビジュアライザ ===")
    print("  [6]: BBox全パラメータを一括入力")
    print("  [7]: 個別パラメータを対話的に微調整")
    print("  [l]: 設定ファイルからBBoxを読み込む")
    print("  [s]: フィルタ確定・保存して終了 (JSONに自動保存)")
    print("  [q]: 保存せずに終了")
    print("  緑色: BBox内 (保持)  灰色: BBox外 (除去)  赤枠: BBox境界")
    print(f"  設定ファイル: {config_path}")
    print("=======================================\n")

    # 初期BBoxがあれば表示
    if region_holder[0] is not None:
        print("初期BBox:")
        print_region(region_holder[0])

    # 初回描画 (初期BBoxが設定済みなら色も反映)
    # draw_geometries_with_key_callbacks はブロッキング呼び出し
    # initial_bbox があれば描画前に色を更新しておく
    if region_holder[0] is not None:
        mask = build_inside_mask(original_points, region_holder[0])
        colors = np.where(
            mask[:, np.newaxis],
            np.array([0.0, 0.8, 0.0]),
            np.array([0.5, 0.5, 0.5]),
        )
        display_pcd.colors = o3d.utility.Vector3dVector(colors)

    geometries = [display_pcd]
    if region_holder[0] is not None:
        init_ls = create_bbox_lineset(region_holder[0])
        bbox_ls_holder[0] = init_ls
        geometries.append(init_ls)

    o3d.visualization.draw_geometries_with_key_callbacks(
        geometries,
        key_to_callback={
            ord("6"): key_set_bbox,
            ord("7"): key_adjust,
            ord("l"): key_load,
            ord("s"): key_confirm,
            ord("q"): key_quit,
        },
        window_name="PCD BBox Filter  [6]:一括設定  [7]:個別調整  [l]:読込  [s]:保存  [q]:終了",
        width=1280,
        height=720,
    )

    if confirmed[0] and region_holder[0] is not None:
        save_bbox_config(region_holder[0], config_path)
        return region_holder[0]
    return None


# ---------------------------------------------------------------------------
# ファイル処理
# ---------------------------------------------------------------------------

def process_files(
    pcd_files: List[str], output_dir: str, region: Tuple[float, ...]
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    xmin, xmax, ymin, ymax, zmin, zmax = region
    print(f"\n処理開始: {len(pcd_files)} ファイル")
    print(f"BBox: X[{xmin}, {xmax}]  Y[{ymin}, {ymax}]  Z[{zmin}, {zmax}]")
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


# ---------------------------------------------------------------------------
# エントリポイント
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PCD BBox Filter: Open3D上でBBoxを指定してエリア外の点群を除去する",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # インタラクティブ (1ファイル)
  python pcd_bbox_filter.py input.pcd output_dir/

  # インタラクティブ (ディレクトリ一括)
  python pcd_bbox_filter.py input_dir/ output_dir/

  # 前回の設定ファイルを使って起動
  python pcd_bbox_filter.py input_dir/ output_dir/ --config bbox.json

  # BBox直接指定 (GUIなしバッチ処理)
  python pcd_bbox_filter.py input_dir/ output_dir/ --bbox -5,5,-3,3,0,2.5
        """,
    )
    parser.add_argument("input", help="入力PCDファイル or PCDファイルが入ったディレクトリ")
    parser.add_argument("output_dir", help="フィルタリング後のPCDを保存するディレクトリ")
    parser.add_argument(
        "--bbox", "-b",
        metavar="xmin,xmax,ymin,ymax,zmin,zmax",
        help="BBox座標を直接指定 (省略時はインタラクティブに指定)",
        default=None,
    )
    parser.add_argument(
        "--config", "-c",
        metavar="PATH",
        help=f"BBox設定ファイルのパス (デフォルト: ./{DEFAULT_CONFIG_NAME})",
        default=None,
    )
    args = parser.parse_args()

    # 設定ファイルパスを決定
    config_path = args.config if args.config else DEFAULT_CONFIG_NAME

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
        # コマンドライン直接指定
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
        # インタラクティブモード
        ref_file = pcd_files[0]
        print(f"\n参照ファイル: {ref_file}")
        ref_pcd = o3d.io.read_point_cloud(ref_file)
        pts = np.asarray(ref_pcd.points)
        print_pcd_stats(pts, os.path.basename(ref_file))

        # 設定ファイルが存在すれば読み込んで初期値として使用
        initial_region: Optional[Tuple[float, ...]] = None
        if os.path.exists(config_path):
            loaded = load_bbox_config(config_path)
            if loaded is not None:
                print(f"\n前回の設定ファイルを読み込みました: {config_path}")
                print_region(loaded)
                initial_region = loaded

        region = interactive_bbox_selection(  # type: ignore[assignment]
            ref_pcd, config_path, initial_region
        )
        if region is None:
            print("BBoxが確定されませんでした。終了します。")
            sys.exit(0)
        print(f"\n確定BBox:")
        print_region(region)

    # フィルタリングと保存
    process_files(pcd_files, args.output_dir, region)


if __name__ == "__main__":
    main()
