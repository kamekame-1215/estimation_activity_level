このプロジェクトのプログラムをパイプラインの手順通りに実行コマンドを列挙します。

## 1. キャリブレーション（センサー位置合わせ）

```bash
python3 adjust_calib_telecognix.py [binディレクトリパス1] [開始フレーム番号1] [binディレクトリパス2] [開始フレーム番号2] [スケール(オプション)]
```

- 例: `python3 adjust_calib_telecognix.py sensor1_dir 0 sensor2_dir 0 1`
- 出力される変換行列をメモしておく [0-cite-0](#0-cite-0) [0-cite-1](#0-cite-1)

## 2. 点群マージ

```bash
python3 merge.py [点群ディレクトリパス1] [点群ディレクトリパス2] [保存先ディレクトリ]
```

- 例: `python3 merge.py sensor1_dir sensor2_dir merged_dir`
- コード内のR1, R2行列をキャリブレーション結果で書き換える必要あり [0-cite-2](#0-cite-2) [0-cite-3](#0-cite-3)

## 3. 領域調整（可視化で領域を決定）

```bash
python3 adjust_region.py [binディレクトリパス] [ファイル名の種類] [初期フレーム番号(オプション)]
```

- 例: `python3 adjust_region.py merged_dir LiDAR 0`
- キー操作: 1/2でフレーム進む、9/0で戻る、6で領域入力 [0-cite-4](#0-cite-4) [0-cite-5](#0-cite-5)

## 4. 領域抽出と平面除去

```bash
python3 region_extraction.py [入力ディレクトリ] [出力ディレクトリ] [--test]
```

- 通常実行: `python3 region_extraction.py merged_dir extracted_dir`
- テストモード（1ファイルのみ可視化）: `python3 region_extraction.py merged_dir extracted_dir --test` [0-cite-6](#0-cite-6)

## 5. ZNCC特徴量抽出

```bash
python3 feature_zncc.py [マージ後点群ディレクトリ] [出力CSVファイル(オプション)]
```

- 例: `python3 feature_zncc.py extracted_dir`
- デフォルト出力: `./pointcloud_feature/feature_ZNCC.csv` [0-cite-7](#0-cite-7)

## 6. 重心特徴量抽出

```bash
python3 feature_centroid.py [マージ後点群ディレクトリ] [出力CSVファイル(オプション)]
```

- 例: `python3 feature_centroid.py extracted_dir`
- デフォルト出力: `./pointcloud_feature/feature_centroid.csv` [0-cite-8](#0-cite-8)

## 7. クラスタリング

```bash
python3 cluster.py [入力CSV] [出力CSV(オプション)] [--plot]
```

- 例: `python3 cluster.py input_data.csv output_data.csv --plot`
- 入力CSVには `ID`, `pr_k`, `score`, `test_score` 列が必要 [0-cite-9](#0-cite-9)

## 8. XGBoost分類（random_forest_activity.py）

```bash
# 1ファイルのみ（学習・テスト分割）
python3 random_forest_activity.py dataset1.csv --drop_cols colA colB --output_csv metrics_xgb.csv

# 2ファイル（学習用・テスト用）
python3 random_forest_activity.py dataset1.csv dataset2.csv --drop_cols1 cA cB --drop_cols2 cX cY --output_csv metrics_xgb.csv
```

- 実際にはXGBoostを使用 [0-cite-10](#0-cite-10)

## 9. XGBoost分類（xgboost_activity.py）

```bash
# 1ファイルのみ
python3 xgboost_activity.py dataset1.csv --drop_cols colA colB --output_csv metrics_xgb.csv

# 2ファイル
python3 xgboost_activity.py dataset1.csv dataset2.csv --drop_cols1 cA cB --drop_cols2 cX cY --output_csv metrics_xgb.csv
```

- random_forest_activity.pyと同様のインターフェース [0-cite-11](#0-cite-11)

**注意**: `data_augmentor.py`はモジュールとして他のスクリプトからインポートされて使用されるため、単独では実行しません [0-cite-12](#0-cite-12) 。
