import os
import glob
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

def default_prefix_func(content):
    """Default prefix function"""
    return f"[PREFIX] {content}"

def process_prompt(prompt_array, question, prefix_func):
    """
    Modify the prompt field and add a prefix.
    
    Args:
        prompt_array (list): Original prompt array
        question (str): Question text
        prefix_func (function): Custom prefix function
    
    Returns:
        list: Modified prompt array
    """
    try:
        new_prompt = [
            {
                "role": item.get("role", "user"),
                "content": prefix_func(question)
            }
            for item in prompt_array
        ]
        return new_prompt
    except Exception as e:
        print(f"❌ Error occurred while processing prompt: {e}")
        return prompt_array  # Return the original data to avoid task failure

def process_parquet_files(input_dir, output_dir, prefix_func=default_prefix_func):
    """
    Automated processing of Parquet files (pure Python implementation)
    
    Args:
        input_dir (str): Input directory path
        output_dir (str): Output directory path
        prefix_func (function): Custom prefix handling function
    """
    try:
        # Recursively find all Parquet files
        file_paths = glob.glob(os.path.join(input_dir, "**/*.parquet"), recursive=True)
        if not file_paths:
            print("⚠️ No Parquet files found")
            return

        for file_path in file_paths:
            # Construct output path
            relative_path = os.path.relpath(file_path, input_dir)
            base_name, _ = os.path.splitext(relative_path)
            output_file = os.path.join(output_dir, f"{base_name}_p.parquet")
            
            # Create output directories
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Read Parquet file
            table = pq.read_table(file_path)
            df = table.to_pandas()
            
            # Validate data structure
            if "prompt" not in df.columns or "question" not in df.columns:
                print(f"⚠️ Skipping file (missing required fields): {file_path}")
                continue
            
            # Add prefix
            df["prompt"] = df.apply(
                lambda row: process_prompt(row["prompt"], row["question"], prefix_func),
                axis=1
            )
            
            # Convert back to Arrow Table
            table = pa.Table.from_pandas(df)
            
            # Save result
            pq.write_table(table, output_file)
            print(f"✅ Processed and saved: {output_file}")
    
    except Exception as e:
        print(f"❌ Processing failed: {e}")

if __name__ == "__main__":
    def my_custom_prefix(question):
        """prefix function"""
        question = question.strip()
        if question[-1] != '?':
            question += '?'
        return f"""
    ## Background Information  
    # Role Definition  
    You are a specialized **Information Retrieval Agent**. Perform reasoning and use the search tool before providing the final answer.
    You should continue searching until all the required information has been retrieved, and then provide the final answer.

    ## Note-Taking Rules  
    When retrieving information enclosed in `<information>`, summarize its content and use the following markers to highlight key or uncertain elements:

    There are two types of markers:
    1. `-` (Uncertainty): Marks ambiguous or uncertain information.  
    Example: `-He picked up Jenny-` (Uncertain who "he" refers to).
    2. `*` (Key Info): Highlights important or critical details.  
    Example: *Built in 1900* (The year is essential).

    ## Format Instructions  
    - Use `<search>Your query</search>` to call the search tool.
    - For each `<information>Search result</information>`, provide a summarized version in `<summary>`, using the above markers to indicate key or uncertain information.
    - Only output the final answer inside `<answer></answer>`. Do not include explanations, reasoning, or extra text.
    - If it's a yes/no question, respond only with `yes` or `no`.
    - Always follow this format strictly.
    - **Answer must be in English. Only English responses will be accepted.**
    Note: No searches allowed after answer submission. So avoid answering when uncertain – verify accuracy thoroughly before answering
    Question: {question}
    """

    input_directory = "./data/"
    output_directory = "./data/"

    process_parquet_files(input_directory, output_directory, my_custom_prefix)
