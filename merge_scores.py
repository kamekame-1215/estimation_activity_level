"""
ExcelのスコアデータとCSV点群特徴量をsubject_id + video_idで結合するプログラム

使い方:
  # 特徴量CSVにsubject_id・video_id列が既にある場合
  python3 merge_scores.py --excel scores.xlsx --features features.csv --output merged.csv

  # 特徴量CSVのID列が "subject_1_video_1" 形式の場合
  python3 merge_scores.py --excel scores.xlsx --features features.csv --split_id

Excel形式:
  シート名: video_1, video_2, video_3, video_4 ...
  各シートの列: subject_id, pr_k, score, test_score (他の列は無視)

特徴量CSV形式(デフォルト):
  列: subject_id, video_id, <特徴量列...>

特徴量CSV形式(--split_id使用時):
  列: ID (例: "subject_1_video_1"), <特徴量列...>

出力:
  特徴量 + スコア(pr_k, score, test_score)が結合されたCSV
  → そのままcluster.pyやxgboost_activity.pyに渡せる形式
"""

import argparse
import os
import sys

import pandas as pd

SCORE_COLS = ["pr_k", "score", "test_score"]


def load_excel_scores(excel_path: str, subject_col: str) -> pd.DataFrame:
    """
    各シート(video_1, video_2...)を読み込み、video_id列を付加して縦結合する。
    """
    xl = pd.ExcelFile(excel_path)
    dfs = []
    for sheet_name in xl.sheet_names:
        df = xl.parse(sheet_name)
        df.columns = [str(c).strip() for c in df.columns]
        if subject_col not in df.columns:
            print(f"  Warning: シート '{sheet_name}' に '{subject_col}' 列がありません。スキップします。")
            continue
        df["video_id"] = sheet_name  # "video_1" など
        dfs.append(df)

    if not dfs:
        raise ValueError("有効なシートが見つかりませんでした。subject_id列の列名を --subject_col で指定してください。")

    return pd.concat(dfs, ignore_index=True)


def split_composite_id(df: pd.DataFrame, id_col: str = "ID") -> pd.DataFrame:
    """
    "subject_1_video_1" 形式のID列を subject_id と video_id に分割する。
    'video' というキーワードが現れる手前までを subject_id とする。
    """
    def parse_id(id_str: str):
        parts = str(id_str).split("_")
        for i, part in enumerate(parts):
            if part == "video":
                subject_id = "_".join(parts[:i])
                video_id = "_".join(parts[i:])
                return subject_id, video_id
        # "video" が見つからない場合はそのままIDをsubject_idとして扱う
        return id_str, ""

    parsed = df[id_col].apply(lambda x: pd.Series(parse_id(x), index=["subject_id", "video_id"]))
    df = df.copy()
    df[["subject_id", "video_id"]] = parsed
    return df


def main():
    parser = argparse.ArgumentParser(
        description="ExcelスコアとCSV点群特徴量をsubject_id + video_idで結合します",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--excel", required=True, help="スコアExcelファイルのパス (.xlsx)")
    parser.add_argument("--features", required=True, help="点群特徴量CSVファイルのパス")
    parser.add_argument("--output", default="merged_dataset.csv", help="出力CSVファイルのパス (デフォルト: merged_dataset.csv)")
    parser.add_argument(
        "--subject_col",
        default="subject_id",
        help="Excelおよび特徴量CSV内のsubject列名 (デフォルト: subject_id)",
    )
    parser.add_argument(
        "--split_id",
        action="store_true",
        help="特徴量CSVのID列が 'subject_1_video_1' 形式の場合に指定",
    )
    args = parser.parse_args()

    # --- 入力ファイルの存在確認 ---
    for path, label in [(args.excel, "Excel"), (args.features, "特徴量CSV")]:
        if not os.path.exists(path):
            print(f"Error: {label}ファイルが見つかりません: {path}")
            sys.exit(1)

    # --- [1] Excel読み込み ---
    print(f"[1/3] Excel読み込み中: {args.excel}")
    scores_df = load_excel_scores(args.excel, args.subject_col)
    video_ids = sorted(scores_df["video_id"].unique().tolist())
    subjects = sorted(scores_df[args.subject_col].unique().tolist())
    print(f"  → {len(scores_df)}行  動画: {video_ids}  被験者数: {len(subjects)}")

    # --- [2] 特徴量CSV読み込み ---
    print(f"[2/3] 特徴量CSV読み込み中: {args.features}")
    features_df = pd.read_csv(args.features)
    features_df.columns = [str(c).strip() for c in features_df.columns]
    print(f"  → {len(features_df)}行  列: {features_df.columns.tolist()}")

    # --split_id の場合はID列をsubject_id / video_idに分割
    if args.split_id:
        if "ID" not in features_df.columns:
            print("Error: --split_id を指定しましたが、特徴量CSVに 'ID' 列がありません。")
            sys.exit(1)
        features_df = split_composite_id(features_df, "ID")
        print(f"  → ID列を subject_id / video_id に分割しました")

    # キー列の存在チェック
    missing_keys = [k for k in [args.subject_col, "video_id"] if k not in features_df.columns]
    if missing_keys:
        print(f"Error: 特徴量CSVに列 {missing_keys} がありません。")
        print(f"  存在する列: {features_df.columns.tolist()}")
        print("  ヒント: 'subject_1_video_1' 形式のID列がある場合は --split_id を指定してください。")
        sys.exit(1)

    # --- [3] マージ ---
    available_score_cols = [c for c in SCORE_COLS if c in scores_df.columns]
    if not available_score_cols:
        print(f"Warning: Excelに {SCORE_COLS} のいずれの列もありません。")
        print(f"  Excel列: {scores_df.columns.tolist()}")

    merge_keys = [args.subject_col, "video_id"]
    scores_slim = scores_df[merge_keys + available_score_cols].drop_duplicates()

    print(f"[3/3] マージ中 (キー: {merge_keys}) ...")
    merged = features_df.merge(scores_slim, on=merge_keys, how="left")

    # マッチしなかった行の警告
    if available_score_cols:
        missing_mask = merged[available_score_cols[0]].isna()
        missing_count = int(missing_mask.sum())
        if missing_count > 0:
            print(f"\nWarning: {missing_count}行でスコアデータが見つかりませんでした:")
            print(merged.loc[missing_mask, merge_keys].to_string(index=False))

    merged.to_csv(args.output, index=False)

    total = len(merged)
    matched = total - (int(merged[available_score_cols[0]].isna().sum()) if available_score_cols else 0)
    print(f"\n結合完了: {matched}/{total}行マッチ → {args.output}")


if __name__ == "__main__":
    main()
