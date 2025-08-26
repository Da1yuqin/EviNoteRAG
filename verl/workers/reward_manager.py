from verl import DataProto
import torch
from verl.utils.reward_score.parser_utils import *
import ray
import time
from verl.utils.entropy.entilement_score import EntailmentModelHolder

# reward func 
from verl.utils.reward_score import qa_train, qa_val

def _select_rm_score_fn(data_source, val_type="em"):
    # train
    if data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']:
        return qa_train.batched_compute_score_em_ver
    # test
    else: 
        type_to_func = {
            "em": qa_val.compute_score_em_batch,
            "f1": qa_val.compute_score_f1_batch,
            "retrival": qa_val.compute_score_retrival_batch,
        }
        if val_type in type_to_func:
            return type_to_func[val_type]
        else:
            raise ValueError(f"Unsupported val_type: {val_type}")


class RewardManager():
    """The reward manager. Batch acceleration
    """

    def __init__(self, tokenizer, num_examine, format_score=0., model_holder=None,) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.format_score = format_score
        # Build entailment model
        self.model_holder = model_holder
        self.entailment_model = None  
        
    def __call__(self, data: DataProto, val_type ="em"):
        if self.entailment_model is None:
            start = time.time()
            if self.model_holder is None:
                raise RuntimeError("model_holder is not set in RewardManager")
            # Load the model into the Actor
            ray.get(self.model_holder.load_model.remote())
            self.entailment_model = "LOADED"  # Only mark as loaded; do not save the model object.
            print(f"[INFO] Model loaded in {time.time() - start:.2f}s")


        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        sequences_list = []
        ground_truth_list = []
        index_mapping = []
        processed_gt_list = []
        data_source_list = []
        summary_list = []


        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode prompt + response
            combined = torch.cat([valid_prompt_ids, valid_response_ids])
            sequences_str = self.tokenizer.decode(combined, skip_special_tokens=False)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']

            # processed_answer
            ground_truth_strings = extract_ground_truth_strings([ground_truth]) 
            processed_gt = extract_question_and_rewrite_fast(sequences_str, ground_truth_strings) # gt list
            summary = extract_last_summary(sequences_str) # Only the text from the last paragraph

            # select score function
            compute_score_fn = _select_rm_score_fn(data_source, val_type)

            # collect input for remote call
            sequences_list.append(sequences_str)
            ground_truth_list.append(ground_truth_strings[0]) # Take "str" from ["str"]
            data_source_list.append(data_source)
            processed_gt_list.append(processed_gt[0])  # Take gt from [gt]
            summary_list.append(summary)
            index_mapping.append((i, valid_response_length))
        

        # Batch invoke the remote function
        future = compute_score_fn.remote(
            solution_strs=sequences_list,
            ground_truths=ground_truth_list,
            summaries=summary_list,
            processed_gts=processed_gt_list,
            data_sources = data_source_list,
            format_score=self.format_score,
            score=1.0,
            retrival_eval_model=self.model_holder
        )
        scores = ray.get(future)  # Retrieve all results

        # Fill reward_tensor
        for (i, length), score in zip(index_mapping, scores):
            reward_tensor[i, length - 1] = score

        return reward_tensor
