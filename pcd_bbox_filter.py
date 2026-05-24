#!/usr/bin/env python3
"""
PCD BBox Filter

Open3D上でバウンディングボックス(BBox)を対話的に指定し、
エリア外の点群を除去して新しいフォルダに保存するツール。

使い方:
    python pcd_bbox_filter.py <入力PCD or ディレクトリ> <出力ディレクトリ>
    python pcd_bbox_filter.py <入力PCD or ディレクトリ> <出力ディレクトリ> --bbox xmin,xmax,ymin,ymax,zmin,zmax
    python pcd_bbox_filter.py <入力PCD or ディレクトリ> <出力ディレクトリ> --config bbox.json

ビジュアライザのキー操作 (リアルタイム操作):
    C: モード切替 (TRANSLATE=BBox移動 ↔ RESIZE=サイズ変更)

    -- 移動 / サイズ変更 --
    U / J : +x / -x
    I / K : +y / -y
    O / L : +z / -z
      TRANSLATE: BBox全体を平行移動
      RESIZE   : BBoxサイズを対称に拡大/縮小

    -- ステップ幅調整 --
    1/2/3/4/5 : ステップ幅 +0.01/+0.1/+1/+10/+100
    0/9/8/7/6 : ステップ幅 -0.01/-0.1/-1/-10/-100

    -- その他 --
    P: 現在のBBoxパラメータと点数を表示
    N: 設定ファイル(JSON)からBBoxを読み込む
    T: ターミナルでBBox全パラメータを一括入力
    S: フィルタリングを確定して保存・終了 (BBoxをJSONに自動保存)
    Q: 保存せずに終了

点群の色:
    緑色: BBox内 (保持される)
    灰色: BBox外 (除去される)
    赤枠: BBox表示 (TRANSLATEモード)
    青枠: BBox表示 (RESIZEモード)
"""

import json
import os
import sys
import glob
import argparse
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import open3d as o3d

# BBoxパラメータ名(インデックス順)
PARAM_NAMES = ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"]
DEFAULT_CONFIG_NAME = "bbox_config.json"


class AdjustMode(Enum):
    TRANSLATE = 1  # BBox全体を平行移動
    RESIZE = 2     # BBoxサイズを対称に拡大/縮小


# ---------------------------------------------------------------------------
# BBox / 点群ユーティリティ
# ---------------------------------------------------------------------------

def create_bbox_lineset(
    region: Tuple[float, ...],
    color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> o3d.geometry.LineSet:
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
    ls.colors = o3d.utility.Vector3dVector(np.tile(list(color), (len(edges), 1)))
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
    move_size: List[float] = [1.0]
    adjust_mode: List[AdjustMode] = [AdjustMode.TRANSLATE]
    confirmed = [False]

    # 表示用点群 (全点保持・色だけ変更)
    display_pcd = o3d.geometry.PointCloud()
    display_pcd.points = o3d.utility.Vector3dVector(original_points)
    display_pcd.paint_uniform_color([0.5, 0.5, 0.5])

    # ------------------------------------------------------------------
    # 内部更新ヘルパー
    # ------------------------------------------------------------------

    def _bbox_color() -> Tuple[float, float, float]:
        return (1.0, 0.0, 0.0) if adjust_mode[0] == AdjustMode.TRANSLATE else (0.0, 0.4, 1.0)

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
            ls = create_bbox_lineset(region_holder[0], color=_bbox_color())
            vis.add_geometry(ls, reset_bounding_box=False)
            bbox_ls_holder[0] = ls

    def _apply_region(vis: o3d.visualization.Visualizer, silent: bool = False) -> None:
        """region_holder[0] を表示に反映する共通処理"""
        region = region_holder[0]
        if region is not None and not silent:
            mask = build_inside_mask(original_points, region)
            n_inside = int(mask.sum())
            n_total = len(original_points)
            rate = 100.0 * n_inside / n_total if n_total > 0 else 0.0
            print(f"  BBox内の点数: {n_inside} / {n_total} ({rate:.1f}%)")
        _refresh_colors(vis)
        _refresh_bbox(vis)
        vis.update_renderer()

    def _init_region_if_none() -> bool:
        """regionがNoneの場合は点群のAABBで初期化する"""
        if region_holder[0] is None:
            if len(original_points) == 0:
                return False
            pts = original_points
            region_holder[0] = (
                float(pts[:, 0].min()), float(pts[:, 0].max()),
                float(pts[:, 1].min()), float(pts[:, 1].max()),
                float(pts[:, 2].min()), float(pts[:, 2].max()),
            )
            print("BBoxを点群のAABBで初期化しました。")
        return True

    # ------------------------------------------------------------------
    # キーコールバック: モード切替 (C)
    # ------------------------------------------------------------------

    def key_change_mode(vis: o3d.visualization.Visualizer) -> bool:
        if adjust_mode[0] == AdjustMode.TRANSLATE:
            adjust_mode[0] = AdjustMode.RESIZE
            print("モード: RESIZE (サイズ変更)  [青枠]")
        else:
            adjust_mode[0] = AdjustMode.TRANSLATE
            print("モード: TRANSLATE (移動)  [赤枠]")
        _refresh_bbox(vis)
        vis.update_renderer()
        return True

    # ------------------------------------------------------------------
    # キーコールバック: 平行移動 / サイズ変更 (U/J I/K O/L)
    # ------------------------------------------------------------------

    def _move(vis: o3d.visualization.Visualizer, axis: int, sign: int) -> bool:
        if not _init_region_if_none():
            return False
        r = list(region_holder[0])
        step = move_size[0] * sign
        if adjust_mode[0] == AdjustMode.TRANSLATE:
            # axis=0:x, 1:y, 2:z  →  インデックス (0,1) / (2,3) / (4,5)
            r[axis * 2]     += step
            r[axis * 2 + 1] += step
        else:  # RESIZE: 対称に拡大/縮小
            r[axis * 2]     -= step   # min 側は逆方向
            r[axis * 2 + 1] += step
        region_holder[0] = tuple(r)  # type: ignore[assignment]
        _apply_region(vis, silent=True)
        return True

    def translate_x_plus(vis):  return _move(vis, 0, +1)
    def translate_x_minus(vis): return _move(vis, 0, -1)
    def translate_y_plus(vis):  return _move(vis, 1, +1)
    def translate_y_minus(vis): return _move(vis, 1, -1)
    def translate_z_plus(vis):  return _move(vis, 2, +1)
    def translate_z_minus(vis): return _move(vis, 2, -1)

    # ------------------------------------------------------------------
    # キーコールバック: ステップ幅調整 (1-5 / 0,9,8,7,6)
    # ------------------------------------------------------------------

    def _change_step(delta: float) -> bool:
        move_size[0] = max(0.0, move_size[0] + delta)
        print(f"move_size = {move_size[0]:.4f}")
        return False

    def inc_001(_): return _change_step(+0.01)
    def inc_01(_):  return _change_step(+0.1)
    def inc_1(_):   return _change_step(+1.0)
    def inc_10(_):  return _change_step(+10.0)
    def inc_100(_): return _change_step(+100.0)
    def dec_001(_): return _change_step(-0.01)
    def dec_01(_):  return _change_step(-0.1)
    def dec_1(_):   return _change_step(-1.0)
    def dec_10(_):  return _change_step(-10.0)
    def dec_100(_): return _change_step(-100.0)

    # ------------------------------------------------------------------
    # キーコールバック: P → パラメータ表示
    # ------------------------------------------------------------------

    def key_print(_: o3d.visualization.Visualizer) -> bool:
        print(f"\n--- [P] 現在のBBox  (move_size={move_size[0]:.4f}, mode={adjust_mode[0].name}) ---")
        if region_holder[0] is None:
            print("  BBox未設定")
        else:
            print_region(region_holder[0])
            mask = build_inside_mask(original_points, region_holder[0])
            n_inside = int(mask.sum())
            n_total = len(original_points)
            rate = 100.0 * n_inside / n_total if n_total > 0 else 0.0
            print(f"  BBox内の点数: {n_inside} / {n_total} ({rate:.1f}%)")
        return False

    # ------------------------------------------------------------------
    # キーコールバック: N → 設定ファイルから読み込み
    # ------------------------------------------------------------------

    def key_load(vis: o3d.visualization.Visualizer) -> bool:
        print(f"\n--- [N] 設定ファイル読み込み: {config_path} ---")
        loaded = load_bbox_config(config_path)
        if loaded is None:
            return True
        region_holder[0] = loaded
        print("読み込んだBBox:")
        print_region(region_holder[0])
        _apply_region(vis)
        return True

    # ------------------------------------------------------------------
    # キーコールバック: T → ターミナルで一括入力
    # ------------------------------------------------------------------

    def key_set_bbox(vis: o3d.visualization.Visualizer) -> bool:
        print("\n--- [T] BBox一括設定 ---")
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
    # キーコールバック: S → 確定・保存
    # ------------------------------------------------------------------

    def key_confirm(vis: o3d.visualization.Visualizer) -> bool:
        if region_holder[0] is None:
            print("BBoxが設定されていません。[T]キーで設定するかキー操作でBBoxを作成してください。")
            return True
        confirmed[0] = True
        vis.close()
        return True

    # ------------------------------------------------------------------
    # キーコールバック: Q → 終了
    # ------------------------------------------------------------------

    def key_quit(vis: o3d.visualization.Visualizer) -> bool:
        confirmed[0] = False
        vis.close()
        return True

    # ------------------------------------------------------------------
    # ビジュアライザ起動
    # ------------------------------------------------------------------

    usage_message = (
        "\n=== PCD BBox Filter ビジュアライザ ===\n"
        "-- マウス操作 --\n"
        "  左ドラッグ: 回転  Ctrl+左ドラッグ/ホイールドラッグ: 平行移動  ホイール: ズーム\n"
        "-- キー操作 --\n"
        "  C: モード切替 TRANSLATE(移動)[赤枠] ↔ RESIZE(サイズ変更)[青枠]\n"
        "  U / J : +x / -x\n"
        "  I / K : +y / -y\n"
        "  O / L : +z / -z\n"
        "  1/2/3/4/5 : ステップ幅 +0.01/+0.1/+1/+10/+100\n"
        "  0/9/8/7/6 : ステップ幅 -0.01/-0.1/-1/-10/-100\n"
        "  P: BBoxパラメータ表示\n"
        "  N: 設定ファイルからBBoxを読み込む\n"
        "  T: ターミナルでBBox全パラメータを一括入力\n"
        "  S: 確定・保存して終了 (JSONに自動保存)\n"
        "  Q: 保存せずに終了\n"
        f"  設定ファイル: {config_path}\n"
        "======================================="
    )
    print(usage_message)

    # 初期BBoxが未設定なら点群のAABBで初期化
    if region_holder[0] is None and len(original_points) > 0:
        pts = original_points
        region_holder[0] = (
            float(pts[:, 0].min()), float(pts[:, 0].max()),
            float(pts[:, 1].min()), float(pts[:, 1].max()),
            float(pts[:, 2].min()), float(pts[:, 2].max()),
        )
        print("初期BBox: 点群のAABBを使用")

    print("\n初期BBox:")
    if region_holder[0] is not None:
        print_region(region_holder[0])

    # 描画前に点群カラーとBBoxを反映
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
        init_ls = create_bbox_lineset(region_holder[0], color=_bbox_color())
        bbox_ls_holder[0] = init_ls
        geometries.append(init_ls)

    o3d.visualization.draw_geometries_with_key_callbacks(
        geometries,
        key_to_callback={
            ord("U"): translate_x_plus,  ord("J"): translate_x_minus,
            ord("I"): translate_y_plus,  ord("K"): translate_y_minus,
            ord("O"): translate_z_plus,  ord("L"): translate_z_minus,
            ord("C"): key_change_mode,
            ord("1"): inc_001,  ord("2"): inc_01,  ord("3"): inc_1,
            ord("4"): inc_10,   ord("5"): inc_100,
            ord("0"): dec_001,  ord("9"): dec_01,  ord("8"): dec_1,
            ord("7"): dec_10,   ord("6"): dec_100,
            ord("P"): key_print,
            ord("N"): key_load,
            ord("T"): key_set_bbox,
            ord("S"): key_confirm,
            ord("Q"): key_quit,
        },
        window_name=(
            "PCD BBox Filter  "
            "C:モード切替  U/J:±x  I/K:±y  O/L:±z  "
            "1-5:step+  0-6:step-  P:表示  N:読込  T:入力  S:保存  Q:終了"
        ),
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