# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import string
import random
import json

import os
import logging
from verl.utils.entropy.entilement_score import compute_retrival_score_fn
import ray
import numpy as np

logging.basicConfig(level=logging.INFO)

def save_results_to_file(solution_str, answer_content, ground_truths, max_score, data_source=None, f1_score = 0.0, llm_judge_score= 0.0, Search_score= 0.0, retrival_eval_model = None):
    """Save the results to a file with a counter."""
    # Add a counter to the generation part; here just read the counter
    try:
        # Create output directory
        os.makedirs(os.path.dirname("./outputs/eval/"), exist_ok=True)
        
        count_file_path = "./outputs/eval/count.txt"
        current_count = 0
        
        # # Handle the counter file, read if it exists (usually it does)
        if os.path.exists(count_file_path):
            with open(count_file_path, 'r', encoding='utf-8') as f:
                count_str = f.read().strip()
                if count_str.isdigit():
                    current_count = int(count_str)
        else:
             # Create a new counter file
            with open(count_file_path, 'w', encoding='utf-8') as f:
                f.write('0')
                logging.info("Created new counter file: count.txt")
        
        # Generate the file name with the counter
        json_file_path = f"./outputs/eval/train_{current_count}.jsonl"
        
        # Save data to the new file
        save_json = {
            "solution_str": solution_str,
            "answer_content": answer_content,
            "ground_truths": ground_truths,
            "score": max_score,
            "f1_score": f1_score,
            "llm_judge_score": llm_judge_score,
            "Search_score": Search_score,
            "data_source":data_source,
        }
        json_line = json.dumps(save_json, ensure_ascii=False)
        
        # Write to the JSONL file
        with open(json_file_path, 'a', encoding='utf-8') as f:
            f.write(json_line + '\n')
            logging.info(f"File saved: {json_file_path}")
        
            
    except (IOError, OSError) as e:
        logging.error(f"File write failed: {e}")
    except Exception as e:
        logging.error(f"Unknown error: {e}")

  

def check_if_mark_exist(text, keyword_list=["*", "#"]):
    pattern = r'<summary>(.*?)</summary>'
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return False
    
    stack = []
    is_empty = True # check if notes exists
    for content in matches:
        if content == "":
            continue
        else:
            for char in content:
                if char in keyword_list:
                    is_empty = False
                    if stack and stack[-1] == char:
                        stack.pop()
                    else:
                        stack.append(char)
    
    return len(stack) == 0 and (not is_empty)




def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    
    # If there are 0 or exactly 1 matches, return None
    if len(matches) <= 1:
        return None
    
    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def compute_score_em(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
    
    if answer is None:
        return 0
    else:
        if em_check(answer, ground_truth['target']):
            return score
        else:
            return format_score


def compute_score_subem(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
    
    if answer is None:
        return 0
    else:
        if subem_check(answer, ground_truth['target']):
            return score
        else:
            return format_score


def f1_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    
    def compute_f1_score(prediction_tokens, golden_tokens):
        common = set(prediction_tokens) & set(golden_tokens)
        num_same = len(common)
        if num_same == 0:
            return 0
        
        precision = num_same / len(prediction_tokens)
        recall = num_same / len(golden_tokens)
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    normalized_prediction = normalize_answer(prediction).split()
    max_f1 = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer).split()
        f1 = compute_f1_score(normalized_prediction, golden_answer)
        # print(f1)
        if f1 > max_f1:
            max_f1 = f1
    
    return max_f1
    

def compute_score_f1(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """
    The scoring function for F1 score.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
    
    if answer is None:
        return 0
    else:
        f1_value = f1_check(answer, ground_truth['target'])
        if f1_value > 0:
            return score * f1_value
        else:
            return format_score



@ray.remote
def batched_compute_score_f1_ver(
    solution_strs, 
    ground_truths, 
    summaries,  
    processed_gts,
    data_sources,
    format_score=0., 
    score=1., 
    retrival_eval_model=None
):
    """
    Batch process multiple samples to avoid single-instance model inference.
    
    Args:
        solution_strs: List[str], multiple decoded texts
        ground_truths: List[str], list of preprocessed ground truth strings
        summaries: List[str], summaries extracted from solution_str
        processed_gts: List[str], preprocessed answers (e.g., normalized answers)
        format_score: float, penalty score for formatting errors
        score: float, base score for correct answers
        retrival_eval_model: Ray remote actor, supports batch_compute_score method
    
    Returns:
        scores: List[float], the final score for each sample
    """
    # Data length check
    assert len(solution_strs) == len(ground_truths) == len(summaries), "All lists must be the same length"

    # Batch inference for retrieval scores
    # batch_retrival_scores = [0.0] * len(solution_strs)
    if retrival_eval_model is not None:
        batch_retrival_scores = ray.get(
            retrival_eval_model.batch_retrival_score_fast.remote(ground_truths = processed_gts, retrival_infos = summaries)
        )

    scores = []
    for i, (solution_str, ground_truth, summary, data_source) in enumerate(zip(solution_strs, ground_truths, summaries, data_sources)):
        answer = extract_solution(solution_str)

        rd_score = 0.
        f1_value = 0.

        if answer is None or summary is None:
            rd_score = 0
        else:
            f1_value = f1_check(answer, ground_truth)
            if f1_value > 0:
                rd_score = score * f1_value
            elif check_if_mark_exist(solution_str):
                rd_score = 0.1
            else:
                rd_score = format_score

        rd_score += batch_retrival_scores[i]
        
        # save_results_to_file(solution_str, answer, ground_truth, rd_score, data_source=data_source,f1_score=f1_value, Search_score=batch_retrival_scores[i])
        if random.randint(1, 64) == 1:
            print(f"-------------- [train] ----------------")
            print(f"Solution string: {solution_str}")
            print(f"Extracted summary: {summary}")
            print(f"Extracted answer: {answer}")
            print(f"Golden answers: {ground_truth}")
            print(f"Score: {rd_score}")
            print(f"F1 value: {f1_value}")
            print(f"Search_score: {batch_retrival_scores[i]}")
            print(f"data_source: {data_source}")
            

        scores.append(rd_score)

    return scores


@ray.remote
def batched_compute_score_em_ver(
    solution_strs, 
    ground_truths, 
    summaries,  
    processed_gts,
    data_sources,
    format_score=0., 
    score=1., 
    retrival_eval_model=None
):
    """
    Batch process multiple samples to avoid single-instance model inference.
    
    Args:
        solution_strs: List[str], multiple decoded texts
        ground_truths: List[str], list of preprocessed ground truth strings
        summaries: List[str], summaries extracted from solution_str
        processed_gts: List[str], preprocessed answers (e.g., normalized answers)
        format_score: float, penalty score for formatting errors
        score: float, base score for correct answers
        retrival_eval_model: Ray remote actor, supports batch_compute_score method
    
    Returns:
        scores: List[float], the final score for each sample
    """
    # Data length check
    assert len(solution_strs) == len(ground_truths) == len(summaries), "All lists must be the same length"

    # Batch inference for retrieval scores
    # batch_retrival_scores = [0.0] * len(solution_strs)
    if retrival_eval_model is not None:
        batch_retrival_scores = ray.get(
            retrival_eval_model.batch_retrival_score_fast.remote(ground_truths = processed_gts, retrival_infos = summaries)
        )

    scores = []
    for i, (solution_str, ground_truth, summary, data_source) in enumerate(zip(solution_strs, ground_truths, summaries, data_sources)):
        answer = extract_solution(solution_str)

        rd_score = 0.

        if answer is None or summary is None:
            rd_score = 0
        else:
            if em_check(answer, ground_truth):
                rd_score = score # 1.0
            elif check_if_mark_exist(solution_str):
                rd_score = 0.1
            else:
                rd_score = format_score # 0.0
        rd_score += batch_retrival_scores[i]
        
        # save_results_to_file(solution_str, answer, ground_truth, rd_score, data_source=data_source, Search_score=batch_retrival_scores[i])
        if random.randint(1, 64) == 1:
            print(f"-------------- [train] ----------------")
            print(f"Solution string: {solution_str}")
            print(f"Extracted summary: {summary}")
            print(f"Extracted answer: {answer}")
            print(f"Golden answers: {ground_truth}")
            print(f"Score: {rd_score}")
            print(f"Search_score: {batch_retrival_scores[i]}")
            print(f"data_source: {data_source}")
            

        scores.append(rd_score)

    return scores
