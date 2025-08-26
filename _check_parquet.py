import pandas as pd

# read data
df = pd.read_parquet('./data_preprocess/data/m_add_test_source_p_selected_final_30Jun.parquet')

# select the first data
row = df.iloc[0]

for col in df.columns:
    print(f"name: {col}")
    print(f"content: {row[col]}\n{'='*40}")
    print()
