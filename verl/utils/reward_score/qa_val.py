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
import requests
from multiprocessing import Pool
import re
import string
import random
import json

import os
import logging
from itertools import islice

from verl.utils import reward_score
logging.basicConfig(level=logging.INFO)
import ray


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
    normalized_prediction = normalized_prediction
    max_f1 = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer).split()
        golden_answer = golden_answer
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
    reward_score = 0.
    f1_value = 0.
    if answer is None:
        return 0
    else:
        f1_value = f1_check(answer, ground_truth['target'])
        if f1_value > 0:
            reward_score =  score * f1_value
        else:
            reward_score =  format_score

    # save_results_to_file(solution_str, answer, ground_truth['target'], f1_value, f1_value)
    if do_print:
        print(f"-------------- [val] ----------------")
        print(f"F1 Value: {f1_value}")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
    
    return reward_score

@ray.remote
def compute_score_f1_batch(
            solution_strs,
            ground_truths,
            summaries,
            processed_gts,
            data_sources,
            format_score=0,
            score=1.0,
            retrival_eval_model=None,
        ):
    """
    The scoring function for F1 score.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    scores = []
    for i, (solution_str, ground_truth,data_source) in enumerate(zip(solution_strs, ground_truths,data_sources)):

        answer = extract_solution(solution_str=solution_str)
        
        reward_score = 0.
        f1_value = 0.
        if answer is None:
            pass # 0.
        else:
            f1_value = f1_check(answer, ground_truth)
            if f1_value > 0:
                reward_score =  score * f1_value
            else:
                reward_score =  format_score

        # save_results_to_file(solution_str, answer, ground_truth, max_score=f1_value, data_source=data_source)
        do_print = random.randint(1, 64) == 1
        if do_print:
            print(f"-------------- [val: f1] ----------------")
            print(f"Solution string: {solution_str}")
            print(f"Extracted answer: {answer}")
            print(f"Golden answers: {ground_truth}")
            print(f"F1 Value: {f1_value}")
            print(f"data_source: {data_source}")

        
        # return reward_score
        scores.append(reward_score)
    return scores

@ray.remote
def compute_score_em_batch(
            solution_strs,
            ground_truths,
            summaries,
            processed_gts,
            data_sources,
            format_score=0,
            score=1.0,
            retrival_eval_model=None,
        ):
    """
    The scoring function for em score.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    scores = []
    for i, (solution_str, ground_truth,data_source) in enumerate(zip(solution_strs, ground_truths,data_sources)):

        answer = extract_solution(solution_str=solution_str)
        
        reward_score = 0.
        if answer is None:
            pass # 0.
        else:
            if em_check(answer, ground_truth): 
                reward_score = score #1.0
            else:
                reward_score =  format_score

        # save_results_to_file(solution_str, answer, ground_truth, max_score=reward_score, data_source=data_source)
        do_print = random.randint(1, 64) == 1
        if do_print:
            print(f"-------------- [val: em] ----------------")
            print(f"Solution string: {solution_str}")
            print(f"Extracted answer: {answer}")
            print(f"Golden answers: {ground_truth}")
            print(f"EM Value: {reward_score}")
            print(f"data_source: {data_source}")

        
        # return reward_score
        scores.append(reward_score)
    return scores


@ray.remote
def compute_score_retrival_batch(
            solution_strs,
            ground_truths,
            summaries,
            processed_gts,
            data_sources,
            format_score=0,
            score=1.0,
            retrival_eval_model=None,
        ):
    """
    The scoring function for retrival score.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    scores = None
    if retrival_eval_model is not None:
        scores = ray.get(
            retrival_eval_model.batch_retrival_score_fast.remote(ground_truths = processed_gts, retrival_infos = summaries)
        )
    print(f"-------------- [val: retrival Done] ----------------")
    return scores



