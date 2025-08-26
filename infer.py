import transformers
import torch
import random
from datasets import load_dataset
import requests

question = "Mike Barnett negotiated many contracts including which player that went on to become general manager of CSKA Moscow of the Kontinental Hockey League?"

# Model ID and device setup
# model_id = "your/model/path"
model_id = '/root/verl_checkpoints/local/actor/global_step_50'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

question = question.strip()
if question[-1] != '?':
    question += '?'
curr_eos = [151645, 151643] # for Qwen2.5 series models
curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'

# Prepare the message
prompt = f"""
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

# Initialize the tokenizer and model
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
model = transformers.AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

# Define the custom stopping criterion
class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        # Encode the string so we have the exact token-IDs pattern
        self.target_ids = [tokenizer.encode(target_sequence, add_special_tokens=False) for target_sequence in target_sequences]
        self.target_lengths = [len(target_id) for target_id in self.target_ids]
        self._tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        # Make sure the target IDs are on the same device
        targets = [torch.as_tensor(target_id, device=input_ids.device) for target_id in self.target_ids]

        if input_ids.shape[1] < min(self.target_lengths):
            return False

        # Compare the tail of input_ids with our target_ids
        for i, target in enumerate(targets):
            if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                return True

        return False

def get_query(text):
    import re
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1]
    else:
        return None

def search(query: str):
    payload = {
            "queries": [query],
            "topk": 3,
            "return_scores": True
        }
    results = requests.post("http://127.0.0.1:8000/retrieve", json=payload).json()['result']
                
    def _passages2string(retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
                        
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        return format_reference

    return _passages2string(results[0])


# Initialize the stopping criteria
target_sequences = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(target_sequences, tokenizer)])

cnt = 0

if tokenizer.chat_template:
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)

print('\n\n################# [Start Reasoning + Searching] ##################\n\n')
print(prompt)
# Encode the chat-formatted prompt and move it to the correct device
while True:
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    attention_mask = torch.ones_like(input_ids)
    
    # Generate text with the stopping criteria
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=1024,
        stopping_criteria=stopping_criteria,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7
    )

    if outputs[0][-1].item() in curr_eos:
        generated_tokens = outputs[0][input_ids.shape[1]:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(output_text)
        break

    generated_tokens = outputs[0][input_ids.shape[1]:]
    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    tmp_query = get_query(tokenizer.decode(outputs[0], skip_special_tokens=True))
    if tmp_query:
        # print(f'searching "{tmp_query}"...')
        search_results = search(tmp_query)
    else:
        search_results = ''

    search_text = curr_search_template.format(output_text=output_text, search_results=search_results)
    prompt += search_text
    cnt += 1
    print(search_text)
