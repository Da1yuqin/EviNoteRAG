import re
import numpy as np

# Step 1: Extract ground truth
def extract_ground_truth_strings(ground_truths):
    result = []
    for item in ground_truths:
        target = item['target']
        if isinstance(target, np.ndarray):
            target_list = target.tolist()
        elif isinstance(target, list):
            target_list = target
        else:
            target_list = [target]
        result.append([str(gt) for gt in target_list]) 
    return result

# Step 2: Extract question and concatenate answer
def extract_question_and_rewrite_fast(text, ground_truths):
    match = re.search(r'question:(.*?)[?？]', text, re.DOTALL | re.IGNORECASE)
    if not match:
        raise ValueError("No content found matching the format 'question:...?'")
    
    question_content = match.group(1).strip()
    
    if isinstance(ground_truths, str):
        ground_truth = [ground_truths]
    results = []
    for ground_truth in ground_truths:
        results.append(
            f"{ground_truth[0]} is the answer to the question:\"{question_content}?\""
        )

    return results

def extract_question_and_rewrite(text, ground_truths):
    match = re.search(r'question:(.*?)[?？]', text, re.DOTALL | re.IGNORECASE)
    if not match:
        raise ValueError("No content found matching the format 'question:...?'")
    
    question_content = match.group(1).strip()
    
    if isinstance(ground_truths, str):
        ground_truth = [ground_truths]
    results = []
    for ground_truth in ground_truths:
        results.append([
            f"{gt} is the answer to the question:\"{question_content}?\""
            for gt in ground_truth
        ])

    return results

def extract_last_summary(text: str) -> str | None:
    """
    Extracts the content of the last completely closed <summary>...</summary> tag.
    Args:
        text (str): Input text, which may contain multiple <summary> tags
        debug (bool): Whether to enable debug log
    Returns:
        str | None: The content inside the last completely closed <summary> tag (with whitespace stripped),
                    or None if no match is found.
    """

    # Define a regex to match all <summary> and </summary>
    tag_pattern = re.compile(r'<summary>|</summary>', re.IGNORECASE)

    # Traverse the text and record positions of tags
    stack = []
    tag_positions = []

    for match in tag_pattern.finditer(text):
        tag = match.group().lower()
        pos = match.start()

        if tag == '<summary>':
            stack.append(pos)
        elif tag in ('</summary>', '<\\summary>'):
            if stack:
                start_pos = stack.pop()
                tag_positions.append((start_pos, pos))

    if not tag_positions:
        return None

    # Take the last completely closed tag pair
    last_start, last_end = tag_positions[-1]
    last_content = text[last_start + len('<summary>'):last_end].strip()

    return last_content


def extract_last_evidence(text: str) -> str | None:
    """
    Extracts the content of the last completely closed <evidence>...</evidence> tag.
    Args:
        text (str): Input text, which may contain multiple <evidence> tags
        debug (bool): Whether to enable debug log
    Returns:
        str | None: The content inside the last completely closed <evidence> tag (with whitespace stripped),
                    or None if no match is found.
    """

    # Define a regex to match all <evidence> and </evidence>
    tag_pattern = re.compile(r'<evidence>|</evidence>', re.IGNORECASE)

    # Traverse the text and record positions of tags
    stack = []
    tag_positions = []

    for match in tag_pattern.finditer(text):
        tag = match.group().lower()
        pos = match.start()

        if tag == '<evidence>':
            stack.append(pos)
        elif tag in ('</evidence>', '<\\evidence>'):
            if stack:
                start_pos = stack.pop()
                tag_positions.append((start_pos, pos))

    if not tag_positions:
        return None

    # Take the last completely closed tag pair
    last_start, last_end = tag_positions[-1]
    last_content = text[last_start + len('<evidence>'):last_end].strip()

    return last_content



if __name__ == "__main__":
    # Example input
    text = "question: Who sang this song? Here are some additional content."
    ground_truth_item = {"target": np.array(["Alan Jackson", "me"])}
    
    ground_truths = extract_ground_truth_strings([ground_truth_item])
    print("ground_truths:", ground_truths)
    # 输出: [['Alan Jackson', 'me']]
    try:
        result = extract_question_and_rewrite_fast(text, ground_truths)
        print("Result:", result)
    except Exception as e:
        print("Error:", e)