import pandas as pd

def add_suffix_to_data_source(input_path, output_path):
    """
    Read the Parquet file, add the suffix '_val' to the data_source field, and save it back in Parquet format.

    Args:
        input_path (str): Path to the input Parquet file
        output_path (str): Path to the output Parquet file
    """
    # Read the Parquet file
    df = pd.read_parquet(input_path)
    
    # Verify that the data_source field exists
    if 'data_source' not in df.columns:
        raise ValueError("The 'data_source' field was not found in the Parquet file")
    
    # Add the '_val' suffix to the data_source column
    df['data_source'] = df['data_source'].astype(str) + '_val'
    
    # Save the modified data to a new Parquet file
    df.to_parquet(output_path, index=False)
    print(f"Data processed successfully and saved to {output_path}")


if __name__ == "__main__":
    input_file = "./data/m_test.parquet"
    output_file = "./data/m_test.parquet"
    add_suffix_to_data_source(input_file, output_file)

