import logging
import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import re
from itertools import chain, accumulate
import ray
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def extract_last_information(text):
    """
    Extract the content within the last <information>...</information> tag.

    Args:
        text (str): Input string

    Returns:
        str or None: The extracted content, or None if not found
    """
    pattern = r"<information>(.*?)</information>"
    matches = re.findall(pattern, text, re.DOTALL)

    if matches:
        return matches[-1]  # return last match
    else:
        return None


class BaseEntailment:
    def save_prediction_cache(self):
        pass

class EntailmentDeberta(BaseEntailment):
    def __init__(self, device=None):
        self.device = device if device is not None else DEVICE
        self.tokenizer = AutoTokenizer.from_pretrained("/mnt/xhunter/xuhuan/Pretrained_Models/deberta-v2-xlarge-mnli")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "/mnt/xhunter/xuhuan/Pretrained_Models/deberta-v2-xlarge-mnli").to(self.device)


        # FIXME:
        # self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xlarge-mnli") 
        # self.model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v2-xlarge-mnli").to(self.device)
        
        

    def check_implication(self, text1, text2, *args, **kwargs):
        inputs = self.tokenizer(
            text1, 
            text2,
            padding=True, 
            truncation=True,
            return_tensors="pt").to(self.device)
        # The model checks if text1 -> text2, i.e. if text2 follows from text1.
        # check_implication('The weather is good', 'The weather is good and I like you') --> 1
        # check_implication('The weather is good and I like you', 'The weather is good') --> 2
        outputs = self.model(**inputs)
        logits = outputs.logits
        # Deberta-mnli returns `neutral` and `entailment` classes at indices 1 and 2.
        largest_index = torch.argmax(F.softmax(logits, dim=1), dim=1)  # pylint: disable=no-member
        prediction = largest_index.detach()
        if os.environ.get('DEBERTA_FULL_LOG', False):
            logging.info('Deberta Input: %s -> %s', text1, text2)
            logging.info('Deberta Prediction: %s', prediction)

        return prediction, logits.detach()

def check_entail(text1, text2, model, entailment_option='bi'):
    # text1, text2: b, n
    implication_1, logits_1 = model.check_implication(text1, text2)
    implication_2, logits_2 = model.check_implication(text2, text1)  # pylint: disable=arguments-out-of-order
    assert (implication_1[0].item() in [0, 1, 2]) and (implication_2[0].item() in [0, 1, 2])

    if entailment_option == 'bi':
        semantically_equivalent = (implication_1 == 2) & (implication_2 == 2)
    elif entailment_option == 'a_entails_b': # text 1 entails text 2
        semantically_equivalent = (implication_1 == 2)
    elif entailment_option == 'b_entails_a': # text 2 entails text 1
        semantically_equivalent = (implication_2 == 2)
    elif entailment_option == 'loose':
        # Check if none of the implications are 0 (contradiction) and not both of them are neutral.
        no_contradiction = (implication_1 != 0) & (implication_2 != 0)
        # neutral
        not_both_neutral = ~((implication_1 == 1) & (implication_2 == 1))
        semantically_equivalent = no_contradiction & not_both_neutral

    entail_logits = {
        'a_entails_b': logits_1,
        'b_entails_a': logits_2,
    }
    return semantically_equivalent, entail_logits


def extract_question_and_combine(text, ground_truth):
    """
    Extract the content after 'question:' from text, and concatenate it with ground_truth.

    Args:
        text (str): The original string, formatted as 'question: ...?'
        ground_truth (list of str): The list of ground truth answers

    Returns:
        str: A string in the format "question ...? answer1 or answer2 ..."
    """
    match = re.search(r'question:(.*?)[?？]', text, re.DOTALL | re.IGNORECASE)
    if not match:
        raise ValueError("No content found matching the pattern 'question:...?'")
    
    question_content = match.group(1).strip()
    
    # Convert the ground truth list into a string joined by 'or'
    ground_truth_str = " or ".join(ground_truth)
    
    result = f"{ground_truth_str} is the answer to \"question:{question_content}?\""

    return result

def extract_question_and_rewrite(text, ground_truth):
    """
    Extract the content after 'question:' from the text and concatenate it with ground_truth.

    Args:
        text (str): Original string, formatted as 'question:...?'.
        ground_truth (list of str): List of ground truth answers.

    Returns:
        str: A string in the format [answer1 is the answer to "question...?"],
             [answer2 is the answer to "question...?"] ...
    """
    # Case-insensitive match for 'question: ...?'
    match = re.search(r'question:(.*?)[?？]', text, re.DOTALL | re.IGNORECASE)
    if not match:
        raise ValueError("No content found matching the pattern 'question:...?'")
    
    question_content = match.group(1).strip()
    
    # Convert the ground truth list into a string joined by 'or'
    for i in range(len(ground_truth)):
        ground_truth[i] = f"{ground_truth[i]} is the answer to \"question:{question_content}?\""

    return ground_truth




def compute_retrival_score_fn(ground_truths, retrival_info, entailment_model):
    """
    Compute the weighted average score using bidirectional entailment logits.

    Args:
        ground_truth (str): The ground truth answer or target sentence
        retrival_info (str): The retrieved information

    Returns:
        float: Matching score (between 0 and 1)
    """
    max_score = 0.0
    for ground_truth in ground_truths:
        if not isinstance(ground_truth, str) or not isinstance(retrival_info, str):
            return 0.0
        
        if not ground_truth.strip() or not retrival_info.strip():
            return 0.0

        try:
           
            semantically_equivalent, entail_logits = check_entail(
                ground_truth, retrival_info, entailment_model, entailment_option='bi'
            )

            # logit
            logits_ab = entail_logits['a_entails_b']  # A ⇒ B 
            logits_ba = entail_logits['b_entails_a']  # B ⇒ A 

            # probs
            probs_ab = F.softmax(logits_ab, dim=-1)
            probs_ba = F.softmax(logits_ba, dim=-1)

            # "entailment"
            entail_prob_ab = probs_ab[0][2].item()  # A ⇒ B 
            entail_prob_ba = probs_ba[0][2].item()  # B ⇒ A 

 
            weighted_score = entail_prob_ba
            # 更新最大分数
            if weighted_score > max_score:
                max_score = weighted_score

        except Exception as e:
            print(f"[Error in compute_retrival_score_fn] {e}")
            return 0.0
        
    return max_score
    


def batch_compute_retrival_score_fn(
    ground_truths: list,
    retrival_infos: list,
    entailment_model
) -> list:
    
    if not ground_truths or not retrival_infos:
        return [0.0] * len(ground_truths)

    scores = []

    for ground_truth, retrival_info in zip(ground_truths, retrival_infos):
        max_score = 0.0
        if not isinstance(ground_truth, str) or not isinstance(retrival_info, str):
            scores.append(0.0)
            continue

        if not ground_truth.strip() or not retrival_info.strip():
            scores.append(0.0)
            continue

        try:
            _, entail_logits = check_entail(
                ground_truth, retrival_info, entailment_model, entailment_option='bi'
            )

            logits_ab = entail_logits['a_entails_b']  # A ⇒ B  logit
            logits_ba = entail_logits['b_entails_a']  # B ⇒ A  logit

            probs_ab = F.softmax(logits_ab, dim=-1)
            probs_ba = F.softmax(logits_ba, dim=-1)

            # "entailment" prob
            entail_prob_ab = probs_ab[0][2].item()  # A ⇒ B 
            entail_prob_ba = probs_ba[0][2].item()  # B ⇒ A 

       
            weighted_score = entail_prob_ba  

            max_score = weighted_score
        except Exception as e:
            print(f"[Error in batch_compute_retrival_score_fn] {e}")
            max_score = 0.0

        scores.append(max_score)

    return scores


def batch_check_entail(texts1: list, texts2: list, model):

    def to_str_list(texts):
        result = []
        for t in texts:
            if isinstance(t, (list, np.ndarray)):
                result.append(' '.join(str(x) for x in t))
            elif t is None:
                result.append('')
            elif isinstance(t, bytes):
                result.append(t.decode('utf-8'))  
            else:
                result.append(str(t))  
        return result

    texts1 = to_str_list(texts1)
    texts2 = to_str_list(texts2)

    MAX_LEN = 512  # max len of texts
    texts1 = [t[:MAX_LEN] if len(t) > 0 and len(t) <= MAX_LEN else "" for t in texts1]
    texts2 = [t[:MAX_LEN] if len(t) > 0 and len(t) <= MAX_LEN else "" for t in texts2]

    valid_pairs = [(t1, t2) for t1, t2 in zip(texts1, texts2) if t1 and t2]
    if not valid_pairs:
        return None, {'b_entails_a': torch.empty(0, 3)}  
    filtered_texts1, filtered_texts2 = zip(*valid_pairs)

    with torch.no_grad():
        inputs_ba = model.tokenizer(texts2, texts1, padding=True, truncation=True, return_tensors="pt").to(model.device)
        outputs_ba = model.model(**inputs_ba)
        logits_ba = outputs_ba.logits

    return None, {
        'b_entails_a': logits_ba
    }
    

def batch_compute_retrival_score_fn(
    ground_truths: list,
    retrival_infos: list,
    entailment_model
) -> list:
    if not ground_truths or not retrival_infos:
        return [0.0] * len(ground_truths)

    ground_truths = [[str(gt).strip() for gt in gts] for gts in ground_truths]
    retrival_infos = [str(info).strip() for info in retrival_infos]

    lengths = [len(gts) for gts in ground_truths]

    flat_ground_truths = list(chain.from_iterable(ground_truths))

    expanded_retrival_infos = []
    for info, count in zip(retrival_infos, lengths):
        expanded_retrival_infos.extend([info] * count)

    try:
        implications_ab, logits_dict = batch_check_entail(flat_ground_truths, expanded_retrival_infos, entailment_model)
    except Exception as e:
        print(f"[Error] Batch inference failed: {e}")
        return [0.0] * len(ground_truths)

    # B ⇒ A 
    probs_ba = F.softmax(logits_dict['b_entails_a'], dim=-1)
    entail_prob_ba = probs_ba[:, 2].cpu().tolist()

    scores = []
    indices = list(accumulate(lengths, initial=0))
    for i in range(len(indices) - 1):
        start, end = indices[i], indices[i + 1]
        scores.append(max(entail_prob_ba[start:end]))

    return scores

def batch_compute_retrival_score_fn_fast_(
    ground_truths: list,
    retrival_infos: list,
    entailment_model,
) -> list:
    
    ground_truths = [str(gt) if not isinstance(gt, str) else gt for gt in ground_truths]
    retrival_infos = [str(info) if not isinstance(info, str) else info for info in retrival_infos]

    implications_ab, logits_dict = batch_check_entail(ground_truths, retrival_infos, entailment_model)


    # B ⇒ A
    probs_ba = F.softmax(logits_dict['b_entails_a'], dim=-1)
    scores = probs_ba[:, 2].cpu().tolist()

    return scores

def batch_compute_retrival_score_fn_fast(
    ground_truths: list,
    retrival_infos: list,
    entailment_model,
    batch_size = 512,
) -> list:
    
    ground_truths = [str(gt) if not isinstance(gt, str) else gt for gt in ground_truths]
    retrival_infos = [str(info) if not isinstance(info, str) else info for info in retrival_infos]

    scores = []
    for i in range(0, len(ground_truths), batch_size):
        batch_ground_truths = ground_truths[i:i+batch_size]
        batch_retrival_infos = retrival_infos[i:i+batch_size]

        implications_ab, logits_dict = batch_check_entail(batch_ground_truths, batch_retrival_infos, entailment_model)

        probs_ba = F.softmax(logits_dict['b_entails_a'], dim=-1)
        scores_batch = probs_ba[:, 2].cpu().tolist()
        scores.extend(scores_batch)

    return scores


@ray.remote(num_gpus=1) 
class EntailmentModelHolder:
    def __init__(self, device="cuda",batch_size = 2048):
        import socket
        self.device = device
        self.model = None
        self.batch_size = batch_size
        print(f"[INFO] EntailmentModelHolder is running on {socket.gethostname()}, PID={os.getpid()}")
        print(f"[INFO] device: {self.device}") 
        print(f"[INFO] batch_size of EntailmentModelHolder: {self.batch_size}") 

    def load_model(self):
        if self.model is not None:
            return
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
            print("[WARNING] CUDA not available. Falling back to CPU.")
        try:
            self.model = EntailmentDeberta(device=self.device)
        except Exception as e:
            print(f"[ERROR] Failed to load model on {self.device}: {e}")
            raise
        # print(f"[Load INFO] torch.cuda.is_available(): {torch.cuda.is_available()}")
        # self.model = EntailmentDeberta(device=self.device)

    def get_model(self):
        if self.model is None:
            raise RuntimeError("Model not loaded yet. Call load_model() first.")
        return self.model
    
    def compute_score(self, text1, text2):
        return self.model.check_implication(text1, text2) 
    
    def batch_compute_score(self, texts1: list, texts2: list) -> list:
        """Batch accelerated version of compute_score"""
        with torch.no_grad():
            inputs = self.model.tokenizer(texts1, texts2, padding=True, truncation=True, return_tensors='pt').to(self.device)
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            scores = probs[:, 2].cpu().tolist()
        return scores
    
    def batch_retrival_score_fast(self, ground_truths: list, retrival_infos: list) -> list:
        """gt：list:[str,...]"""
        return batch_compute_retrival_score_fn_fast(ground_truths, retrival_infos, self.model, batch_size = self.batch_size)

    def batch_retrival_score(self, ground_truths: list, retrival_infos: list) -> list:
        """gt：list:[[str, ..., str], ...]"""
        return batch_compute_retrival_score_fn(ground_truths, retrival_infos, self.model)



if __name__ == "__main__":
    entailment_model = EntailmentDeberta(device="cuda" if torch.cuda.is_available() else "cpu")

    # Sample
    # ground_truths = [["Jupiter", "Mercury"], ["Sun"], ["Venus", "Mars", "Saturn"], ["63 is answer to the question \"highest score of michael jordan in one game?\""]]
    # retrival_infos = ["Mercury and Jupiter", "another retrieval result 2", "third retrieval result 3", ["The highest score of Michael Jordan in one game is 63 points, which he scored against the Boston Celtics in a playoff game on April 20, 1986. This achievement is considered one of his most notable in his career. The information provided does not specify the exact game in which he scored 63 points in the regular season, but it is clear it occurred in the playoffs. No other specific individual game score is mentioned with absolute certainty."]]

    retrival_infos = ['The largest number of former colonies became independent states during the late 18th and early to mid-19th centuries. The information is accurate regarding the inception of independence starting in the late 1700s with the US in 1776, and includes historical references to the independence movements throughout the Americas. It mentions the process of peaceful withdrawal post-World War II but notes ongoing colonies until the 19th century.', '`Reverend Tim Tom in the middle is played by Paul Hipp. He appeared in the show from 2009 until 2018. Details about his other roles and recent film projects are provided but they are not pertinent to the question asked.`', "Based on the information retrieved, Sherrod Brown is running for the U.S. Senate from Ohio. He won the re-election in the November 6, 2018 election against Jim Renacci, the Republican nominee, with 53% of the vote. Other names mentioned in the context, such as Michael Gibbons and Melissa Ackison, were also part of the election but did not win the nomination. The certainty lies in Brown's candidacy and victory, while the other candidates are just noted as having been in the race.", 'The new season of The Six, known as the second season, premiered on May 28, 2018. It was a ten-episode season. -Information regarding when the subsequent seasons would air is lacking-.', 'H. Jon Benjamin voices Bob Belcher in "Bob\'s Burgers". Benjamin, an American actor, voice actor, and comedian, is known for his work in various animated shows, including voicing Bob Belcher in "Bob\'s Burgers".', 'Mercury (Hg) is one of the metals that is liquid at room temperature. Another example is caesium (Cs) with a melting point of 28.5 °C (83.3 °F), and rubidium (Rb) with 39 °C (102 °F). There are other metal alloys that are liquid at room temperature, such as NaK, a sodium-potassium metal alloy, and galinstan, a fusible alloy liquid. The standard liquid metal at room temperature in typical applications is often referred to as a gallium-based alloy, which has a lower vapor pressure and is less toxic.', 'The first month of the Jewish calendar is Tishrei according to most rabbinic traditions, and it marks the beginning of the civil year. The actual agricultural new year, indicating the start of spring, is considered to be in 1 Nisan, according to some Christian and Karaite sources. Some traditions, including Karaite, use the ripening of barley to determine the exact start of 1 Nisan, leading to potential seasonal drift.', 'Latin phrase "Favete linguis!" means "Facilitate [the ritual acts] with your tongues," which translates to "hold your tongue." Key usage is in ancient rituals to avoid unwanted words. The phrase has been used by several notable ancient Roman authors. The longer phrase "shut up your mouth" likely originated from a similar context, where it meant to close a business or physically stop a person from speaking, though its exact etymology for the context of the question remains unclear.', 'The virus can be present in the skin, blood, and respiratory tract during the initial infection. After recovery, it remains dormant in sensory nerve ganglia, capable of reactivation as shingles later in life.', '*Slime Time Live* is the name of the Nickelodeon show that involved slime. Information about whether "Slime Time Live" specifically involved slime directly is ambiguous, but its association with Nickelodeon and the production of slimes suggests the show probably did involve slime. However, the specific name of the show that was a Slime-themed television series on Nickelodeon is *Slime Time Live*.']*400
    ground_truths = ['the 1760s and early 1770s is the answer to the question:"during which time period did the largest number of former colonies become independent states?"', 'Hipp is the answer to the question:"who plays reverend tim tom in the middle?"', 'Bruce Jaynes (L) is the answer to the question:"who is running for us senate from ohio?"', 'May 28, 2018 is the answer to the question:"when does the new season of six come on?"', 'H. Jon Benjamin is the answer to the question:"who does the voice for bob in bob\'s burgers?"', 'Galinstan is the answer to the question:"an example of a metal which is liquid at room temperature is?"', 'Tishrei is the answer to the question:"what is the 1st month of the jewish calendar?"', 'si tacuisses, philosophus mansisses is the answer to the question:"if you had kept your mouth shut latin?"', 'nerve tissues is the answer to the question:"where does chicken pox live in the body?"', 'Figure It Out is the answer to the question:"what was the name of the nickelodeon slime show?"']*400
    scores = batch_compute_retrival_score_fn_fast(ground_truths, retrival_infos, entailment_model) # 0.96

    print(scores)  

    
    

    
