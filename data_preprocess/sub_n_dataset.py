import os
import pyarrow.parquet as pq
import pyarrow as pa

def create_subset_parquet(input_path, output_path, n=2048):
    """
    Extract the first n rows from a Parquet file and save them as a new file.
    
    Args:
        input_path (str): Input Parquet file path
        output_path (str): Output Parquet file path
        n (int): Number of rows to extract
    """
    try:
        # 1. Read the Parquet file
        table = pq.read_table(input_path)
        total_rows = table.num_rows
        print(f"📊 Original file contains {total_rows} rows")

        # 2. Take the first n rows (operate directly on Arrow Table)
        if total_rows <= n:
            limited_table = table  # If rows are less than or equal to n, use the original data
            print("⚠️ Original data rows are less than or equal to n, outputting all data")
        else:
            limited_table = table.slice(0, n)  # Efficient slicing
            print(f"✂️ Extracted the first {n} rows")

        # 3. Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 4. Save as a new Parquet file (merge into a single file)
        pq.write_table(limited_table, output_path)
        print(f"✅ Saved to: {output_path}")
    
    except Exception as e:
        print(f"❌ Processing failed: {e}")



if __name__ == "__main__":
    # Input and output paths
    input_path = "./data_addPrompt/add_test_source_p_selected_final_27Jun.parquet"
    output_path = "./data_addPrompt/_test_nqp600_p.parquet"
    n = 600  # Number of rows to extract

    # Start processing
    create_subset_parquet(input_path, output_path, n)

