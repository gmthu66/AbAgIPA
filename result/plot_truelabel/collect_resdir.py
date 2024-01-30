"""输入保存各个metric方法的文件夹, 从中收集各项数据"""
import os
import os.path as osp
import pandas as pd
res_dir = osp.join(osp.dirname(__file__), 'metric_csv')


def extract_columns_and_save(folder_path):
    # 获取文件夹中所有 CSV 文件的列表
    csv_files = [fpath for fpath in os.listdir(folder_path) if fpath.endswith('.csv') and 'metric' in fpath]
    cols = pd.read_csv(osp.join(folder_path, csv_files[0])).columns
    for col in cols:
        col_df = pd.DataFrame()
        for csv_file in csv_files:
            file_path = os.path.join(folder_path, csv_file)
            # 获取文件名（不带后缀）
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            # 从 CSV 文件中读取数据，以第一行作为列名
            df = pd.read_csv(file_path)
            if col in df.columns:
                new_column_name = f"{file_name}_{col}"
                # 将每一列重命名为文件名，并保存到新的 CSV 文件
                # for column in df.columns:
                col_df[new_column_name] = df[col]

        # 保存新的 DataFrame 到 CSV 文件
        output_file_path = os.path.join(folder_path, f"{col}.csv")
        col_df.to_csv(output_file_path, index=False)
        print(f"Processed all {col} -> {output_file_path}")


if __name__ == "__main__":
    # res_dir = '/data/gm_data/AbAtInteraction/AbAtIPA/abatInter_SCA/SabDab/trained_models/res_dir'
    extract_columns_and_save(res_dir)
