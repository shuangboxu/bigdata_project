import pandas as pd
import pandas.testing as pdt
import os

def test_save_and_load_csv(tmp_path):
    # 创建 sample DataFrame
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": ["x", "y", "z"]
    })
    file_path = tmp_path / "sample.csv"
    # 保存
    df.to_csv(file_path, index=False)
    assert file_path.exists()

    # 读取
    df2 = pd.read_csv(file_path)
    # 比较
    pdt.assert_frame_equal(df, df2)
