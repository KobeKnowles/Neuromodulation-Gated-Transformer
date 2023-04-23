from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys

## some code is taken from https://sheng-z.github.io/ReCoRD-explorer/
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def evaluate(answer_data, pred_data):
    f1 = exact_match = total = 0
    correct_ids = []
    incorrect_ids =  []
    for example in answer_data:
        for qa in example['qas']:
            total += 1
            #if qa['idx'] not in predictions:
                #message = 'Unanswered question {} will receive score 0.'.format(qa['idx'])
                #print(message, file=sys.stderr)
                #continue
            prediction = next((item for item in pred_data if int(item["idx"]) == int(qa['idx'])), None)["label"]
            assert len([item for item in pred_data if int(item["idx"]) == int(qa['idx'])]) == 1, f"The number of ids" \
                                                                                            f" equal to {qa['idx']}" \
                                                                                            f" is " \
                                                                                            f"greater than 1!"

            ground_truths = list(map(lambda x: x['text'], qa['answers']))
            print(f"prediction: {prediction}")
            print(f"ground_truths: {ground_truths}")
            #print(f"qa: {qa}")
            #prediction = predictions[qa['id']]

            _exact_match = metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
            if int(_exact_match) == 1:
                correct_ids.append(qa['idx'])
            else: incorrect_ids.append(qa["idx"])
            exact_match += _exact_match

            f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    print('* Exact_match: {}\n* F1: {}'.format(exact_match, f1))

    return {'exact_match': exact_match, 'f1': f1}, correct_ids, incorrect_ids

def save_results_ReCoRD(save_filepath, metrics, i):

    with open(save_filepath, "w") as f:
        f.write(f"epoch: {i}\n"
                f"f1-score: {metrics['f1']}\n"
                f"exact-match: {metrics['exact_match']}")

def ReCoRD_return_correct_incorrect_ids(pred_file, answer_file="/large_data/SuperGlue/ReCoRD/val.jsonl"):

    with open(pred_file, "r") as f:
        pred_data = [json.loads(line) for line in f]  # {"idx":1, "label":"Blah"}

    with open(answer_file, "r") as f:
        answer_data = [json.loads(line) for line in f]

    _, correct_ids, incorrect_ids = evaluate(answer_data, pred_data)
    return correct_ids, incorrect_ids
'''
if __name__ == "__main__":

    #iteration = 100000
    #exp = 1
    #type_ = "gating-end"
    for type_ in ["gating-end", "no-gating-end", "no-gating-no-extra-layers"]:
        for exp in range(1,4):
            for iteration in [50000, 100000, 150000, 200000]:

                dataset = "ReCoRD"
                pred_file = "/data/kkno604/NGT_experiments_updated/general_experiments/"+type_+"/exp"+str(exp)+"/Results/" \
                            "get_results_in_domain/"+dataset+"/prediction_files/iteration_fixed"+str(iteration)+".jsonl"

                answer_file = "/large_data/SuperGlue/"+dataset+"/val.jsonl"

                save_filepath = "/data/kkno604/NGT_experiments_updated/general_experiments/"+type_+"/exp"+str(exp)+"/Results/" \
                                "get_results_in_domain/"+dataset+"/F1/iteration"+str(iteration)+".jsonl"

                with open(pred_file, "r") as f:
                    pred_data = [json.loads(line) for line in f]  # {"idx":1, "label":"Blah"}

                with open(answer_file, "r") as f:
                    answer_data = [json.loads(line) for line in f]

                _, correct_ids, incorrect_ids = evaluate(answer_data, pred_data)

                save_results_ReCoRD(save_filepath, metrics, iteration)
'''