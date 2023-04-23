import json

import sys
sys.path.append("../..")
from data_loaders.MPE import *

def compare_two_jsonl_files_MultiRC_F1a(pred_file, answer_file):

    with open(pred_file, "r") as f:
        pred_data = [json.loads(line) for line in f]  # {"idx":1, "label":0}

    with open(answer_file, "r") as f:
        answer_data = [json.loads(line) for line in f]  # {..., "idx":0, "label":false}

    assert len(pred_data) == len(answer_data), f"The prediction answer set ({len(pred_data)}) and answer answer set " \
                                               f"({len(answer_data)}) do not match in length"

    pred_labels = []
    #print(f"pred_data[0]: {pred_data[0]}")

    for line_dict in pred_data: # prediction is a dict:

        passage_questions = line_dict["passage"]["questions"] # this is a list of dicts with an "idx" and "answers" label

        for dict_answers in passage_questions:
            #print(f"dict_answers: {dict_answers}")

            for dict_labels in dict_answers["answers"]:
                #print(f"dict_labels: {dict_labels}")

                pred_labels.append(dict_labels)
    print("pred_labels:", len(pred_labels))

    answer_labels = []
    for line_dict in answer_data:  # prediction is a dict:

        passage_questions = line_dict["passage"]["questions"]  # this is a list of dicts with an "idx" and "answers" label

        for dict_answers in passage_questions:
            #print(f"dict_answers: {dict_answers}")
            for dict_labels in dict_answers["answers"]:

                answer_labels.append(dict_labels)

    print(f"answer_labels: {len(answer_labels)}")

    assert len(pred_labels) == len(answer_labels), f"The lengths of pred_labels and answer_labels are not equal."

    tp, fp, fn, tn = 0, 0, 0, 0

    for i in range(len(pred_labels)):

        if pred_labels[i]["idx"] == answer_labels[i]["idx"]:
            if pred_labels[i]["label"] == 1:
                if pred_labels[i]["label"] == answer_labels[i]["label"]: tp += 1
                else: fp += 1
            else: # label == 0
                if pred_labels[i]["label"] == answer_labels[i]["label"]: tn += 1
                else: fn += 1

        else: raise Exception(f"The idx values should be equal!")

    f1a_score = get_f1_score(tp=tp, fp=fp, fn=fn, tn=tn)

    print(f"F1a-score: {f1a_score}")

    return f1a_score

def compare_two_jsonl_files_MultiRC_F1a_2(pred_file, answer_file):

    with open(pred_file, "r") as f:
        pred_data = [json.loads(line) for line in f]  # {"idx":1, "label":0}

    with open(answer_file, "r") as f:
        answer_data = [json.loads(line) for line in f]  # {..., "idx":0, "label":false}

    assert len(pred_data) == len(answer_data), f"The prediction answer set ({len(pred_data)}) and answer answer set " \
                                               f"({len(answer_data)}) do not match in length"

    pred_labels = []
    #print(f"pred_data[0]: {pred_data[0]}")

    for line_dict in pred_data: # prediction is a dict:

        passage_questions = line_dict["passage"]["questions"] # this is a list of dicts with an "idx" and "answers" label

        for dict_answers in passage_questions:
            #print(f"dict_answers: {dict_answers}")

            for dict_labels in dict_answers["answers"]:
                #print(f"dict_labels: {dict_labels}")

                pred_labels.append(dict_labels)
    print("pred_labels:", len(pred_labels))

    answer_labels = []
    for line_dict in answer_data:  # prediction is a dict:

        passage_questions = line_dict["passage"]["questions"]  # this is a list of dicts with an "idx" and "answers" label

        for dict_answers in passage_questions:
            #print(f"dict_answers: {dict_answers}")
            for dict_labels in dict_answers["answers"]:

                answer_labels.append(dict_labels)

    print(f"answer_labels: {len(answer_labels)}")

    assert len(pred_labels) == len(answer_labels), f"The lengths of pred_labels and answer_labels are not equal."

    # taken from https://github.com/CogComp/multirc/blob/master/multirc_materials/multirc_measures.py and modified.
    agreementCount = 0
    correctCount = 0
    predictCount = 0

    for i in range(len(pred_labels)):

        if pred_labels[i]["idx"] == answer_labels[i]["idx"]:

            if answer_labels[i]["label"] == 1: correctCount += 1

            if pred_labels[i]["label"] == 1: predictCount += 1

            if answer_labels[i]["label"] == 1 and pred_labels[i]["label"] == 1: agreementCount += 1

        else: raise Exception(f"The idx values should be equal!")

    #f1a_score = get_f1_score(tp=tp, fp=fp, fn=fn, tn=tn)
    p1 = (1.0 * agreementCount / predictCount) if predictCount > 0.0 else 1.0
    r1 = (1.0 * agreementCount / correctCount) if correctCount > 0.0 else 1.0
    f1a_score = 2 * r1 * p1 / (p1 + r1)

    print(f"F1a-score: {f1a_score}")

    return f1a_score

def get_f1_score(tp, fp, fn, tn):

    try:
        precision = tp/(tp+fp) # (relevance among the retrieved instances)
        recall = tp/(tp+fn) # (relevant instances that were retrieved).

        # f1_function = lambda precision, recall: (2 * precision * recall) / (precision + recall)
        return (2 * precision * recall) / (precision + recall)
    except ZeroDivisionError as e:
        print(f"Error: {e}!\n"
              f"tp: {tp}\tfp: {fp}\t"
              f"fn: {fn}\ttn: {tn}")
        print("returning zero!")
        return 0

def compare_two_jsonl_files_f1_unweighted_CB(pred_file, answer_file):

    with open(pred_file, "r") as f:
        pred_data = [json.loads(line) for line in f] # e.g., {"idx":1, "label":0}

    with open(answer_file, "r") as f:
        answer_data = [json.loads(line) for line in f] # e.g., {..., "idx":0, "label":false}

    entail_tp = 0
    entail_fp = 0
    entail_tn = 0
    entail_fn = 0
    entail_f1_score = None

    neutral_tp = 0
    neutral_fp = 0
    neutral_tn = 0
    neutral_fn = 0
    neutral_f1_score = None

    contradiction_tp = 0
    contradiction_fp = 0
    contradiction_tn = 0
    contradiction_fn = 0
    contradiction_f1_score = None

    assert len(pred_data) == len(answer_data), f"The prediction answer set ({len(pred_data)}) and answer answer set " \
                                               f"({len(answer_data)}) do not match in length"

    for ans_type in ["entailment","neutral","contradiction"]:
        for prediction in pred_data:  # prediction is a dict:

            # find a line with
            pred_idx = prediction["idx"]
            answer_dict = next((item for item in answer_data if int(item["idx"]) == int(pred_idx)), None)
            assert len([item for item in answer_data if int(item["idx"]) == int(pred_idx)]) == 1
            if answer_dict is None: raise Exception(f"The id {int(pred_idx)} has no associated answer index!")

            if prediction["label"] == ans_type: # prediction = True.
                if answer_dict["label"] == prediction["label"]: # tp: predict a class correctly.
                    if ans_type == "entailment":
                        entail_tp += 1
                    elif ans_type == "neutral":
                        neutral_tp += 1
                    elif ans_type == "contradiction":
                        contradiction_tp += 1
                else: # fp: predict a class incorrectly.
                    if ans_type == "entailment":
                        entail_fp += 1
                    elif ans_type == "neutral":
                        neutral_fp += 1
                    elif ans_type == "contradiction":
                        contradiction_fp += 1

            if prediction["label"] != ans_type: # prediction = False
                if answer_dict["label"] == ans_type: # get false negatives.
                    if ans_type == "entailment":
                        entail_fn += 1
                    elif ans_type == "neutral":
                        neutral_fn += 1
                    elif ans_type == "contradiction":
                        contradiction_fn += 1
                else: # get true negatives.
                    if ans_type == "entailment":
                        entail_tn += 1
                    elif ans_type == "neutral":
                        neutral_tn += 1
                    elif ans_type == "contradiction":
                        contradiction_tn += 1

        if ans_type == "entailment": entail_f1_score = get_f1_score(tp=entail_tp, fp=entail_fp,
                                                                    fn=entail_fn, tn=entail_tn)
        elif ans_type == "neutral": neutral_f1_score = get_f1_score(tp=neutral_tp, fp=neutral_fp,
                                                                    fn=neutral_fn, tn=neutral_tn)
        elif ans_type == "contradiction": contradiction_f1_score = get_f1_score(tp=contradiction_tp, fp=contradiction_fp,
                                                                                fn=contradiction_fn, tn=contradiction_tn)

    f1_score = (entail_f1_score+neutral_f1_score+contradiction_f1_score) / 3
    print(f"entail_f1_score: {entail_f1_score}\n"
          f"neutral_f1_score: {neutral_f1_score}\n"
          f"contradiction_f1_score: {contradiction_f1_score}\n"
          f"macro_average_f1_score: {f1_score}")

    return entail_f1_score, neutral_f1_score, contradiction_f1_score, f1_score

def compare_two_jsonl_files_f1_unweighted_MPE(pred_file, answer_file):

    with open(pred_file, "r") as f:
        pred_data = [json.loads(line) for line in f] # e.g., {"idx":1, "label":0}

    #with open(answer_file, "r") as f:
    #    answer_data = [json.loads(line) for line in f] # e.g., {..., "idx":0, "label":false}
    dataLoader = MPELoader(filepath=answer_file, tokenizer=None, is_test=False, mode="softmax")
    answer_data = dataLoader.processed_data

    entail_tp = 0
    entail_fp = 0
    entail_tn = 0
    entail_fn = 0
    entail_f1_score = None

    neutral_tp = 0
    neutral_fp = 0
    neutral_tn = 0
    neutral_fn = 0
    neutral_f1_score = None

    contradiction_tp = 0
    contradiction_fp = 0
    contradiction_tn = 0
    contradiction_fn = 0
    contradiction_f1_score = None

    assert len(pred_data) == len(answer_data), f"The prediction answer set ({len(pred_data)}) and answer answer set " \
                                               f"({len(answer_data)}) do not match in length"

    for ans_type in ["entailment","neutral","contradiction"]:
        for prediction in pred_data:  # prediction is a dict:

            # find a line with
            pred_idx = prediction["idx"]
            answer_dict = next((item for item in answer_data if int(item["idx"]) == int(pred_idx)), None)
            assert len([item for item in answer_data if int(item["idx"]) == int(pred_idx)]) == 1
            if answer_dict is None: raise Exception(f"The id {int(pred_idx)} has no associated answer index!")

            if prediction["label"] == ans_type: # prediction = True.
                if answer_dict["gold_label"] == prediction["label"]: # tp: predict a class correctly.
                    if ans_type == "entailment":
                        entail_tp += 1
                    elif ans_type == "neutral":
                        neutral_tp += 1
                    elif ans_type == "contradiction":
                        contradiction_tp += 1
                else: # fp: predict a class incorrectly.
                    if ans_type == "entailment":
                        entail_fp += 1
                    elif ans_type == "neutral":
                        neutral_fp += 1
                    elif ans_type == "contradiction":
                        contradiction_fp += 1

            if prediction["label"] != ans_type: # prediction = False
                if answer_dict["gold_label"] == ans_type: # get false negatives.
                    if ans_type == "entailment":
                        entail_fn += 1
                    elif ans_type == "neutral":
                        neutral_fn += 1
                    elif ans_type == "contradiction":
                        contradiction_fn += 1
                else: # get true negatives.
                    if ans_type == "entailment":
                        entail_tn += 1
                    elif ans_type == "neutral":
                        neutral_tn += 1
                    elif ans_type == "contradiction":
                        contradiction_tn += 1

        if ans_type == "entailment": entail_f1_score = get_f1_score(tp=entail_tp, fp=entail_fp,
                                                                    fn=entail_fn, tn=entail_tn)
        elif ans_type == "neutral": neutral_f1_score = get_f1_score(tp=neutral_tp, fp=neutral_fp,
                                                                    fn=neutral_fn, tn=neutral_tn)
        elif ans_type == "contradiction": contradiction_f1_score = get_f1_score(tp=contradiction_tp, fp=contradiction_fp,
                                                                                fn=contradiction_fn, tn=contradiction_tn)

    f1_score = (entail_f1_score+neutral_f1_score+contradiction_f1_score) / 3
    print(f"entail_f1_score: {entail_f1_score}\n"
          f"neutral_f1_score: {neutral_f1_score}\n"
          f"contradiction_f1_score: {contradiction_f1_score}\n"
          f"macro_average_f1_score: {f1_score}")

    return entail_f1_score, neutral_f1_score, contradiction_f1_score, f1_score

def compare_two_jsonl_files_f1_unweighted_SNLI(pred_file, answer_file):

    with open(pred_file, "r") as f:
        pred_data = [json.loads(line) for line in f] # e.g., {"idx":1, "label":0}

    with open(answer_file, "r") as f:
        answer_data = [json.loads(line) for line in f] # e.g., {..., "idx":0, "label":false}

    entail_tp = 0
    entail_fp = 0
    entail_tn = 0
    entail_fn = 0
    entail_f1_score = None

    neutral_tp = 0
    neutral_fp = 0
    neutral_tn = 0
    neutral_fn = 0
    neutral_f1_score = None

    contradiction_tp = 0
    contradiction_fp = 0
    contradiction_tn = 0
    contradiction_fn = 0
    contradiction_f1_score = None

    assert len(pred_data) == len(answer_data), f"The prediction answer set ({len(pred_data)}) and answer answer set " \
                                               f"({len(answer_data)}) do not match in length"

    for ans_type in ["entailment","neutral","contradiction"]:
        for prediction in pred_data:  # prediction is a dict:

            # find a line with
            pred_idx = prediction["idx"]
            answer_dict = answer_data[pred_idx]
            #answer_dict = next((item for item in answer_data if int(item["idx"]) == int(pred_idx)), None)
            #assert len([item for item in answer_data if int(item["idx"]) == int(pred_idx)]) == 1
            #if answer_dict is None: raise Exception(f"The id {int(pred_idx)} has no associated answer index!")

            if prediction["label"] == ans_type: # prediction = True.
                if answer_dict["gold_label"] == prediction["label"]: # tp: predict a class correctly.
                    if ans_type == "entailment":
                        entail_tp += 1
                    elif ans_type == "neutral":
                        neutral_tp += 1
                    elif ans_type == "contradiction":
                        contradiction_tp += 1
                else: # fp: predict a class incorrectly.
                    if ans_type == "entailment":
                        entail_fp += 1
                    elif ans_type == "neutral":
                        neutral_fp += 1
                    elif ans_type == "contradiction":
                        contradiction_fp += 1

            if prediction["label"] != ans_type: # prediction = False
                if answer_dict["gold_label"] == ans_type: # get false negatives.
                    if ans_type == "entailment":
                        entail_fn += 1
                    elif ans_type == "neutral":
                        neutral_fn += 1
                    elif ans_type == "contradiction":
                        contradiction_fn += 1
                else: # get true negatives.
                    if ans_type == "entailment":
                        entail_tn += 1
                    elif ans_type == "neutral":
                        neutral_tn += 1
                    elif ans_type == "contradiction":
                        contradiction_tn += 1

        if ans_type == "entailment": entail_f1_score = get_f1_score(tp=entail_tp, fp=entail_fp,
                                                                    fn=entail_fn, tn=entail_tn)
        elif ans_type == "neutral": neutral_f1_score = get_f1_score(tp=neutral_tp, fp=neutral_fp,
                                                                    fn=neutral_fn, tn=neutral_tn)
        elif ans_type == "contradiction": contradiction_f1_score = get_f1_score(tp=contradiction_tp, fp=contradiction_fp,
                                                                                fn=contradiction_fn, tn=contradiction_tn)

    f1_score = (entail_f1_score+neutral_f1_score+contradiction_f1_score) / 3
    print(f"entail_f1_score: {entail_f1_score}\n"
          f"neutral_f1_score: {neutral_f1_score}\n"
          f"contradiction_f1_score: {contradiction_f1_score}\n"
          f"macro_average_f1_score: {f1_score}")

    return entail_f1_score, neutral_f1_score, contradiction_f1_score, f1_score