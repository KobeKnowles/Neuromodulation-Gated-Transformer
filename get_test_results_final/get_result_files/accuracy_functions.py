import json

import sys
sys.path.append("../..")
from data_loaders.MPE import *
from data_loaders.RACE import *
from data_loaders.DREAM import *
from data_loaders.ReClor import *
from data_loaders.StrategyQA import *

def compare_two_jsonl_files_MultiRC_ACC(pred_file, answer_file):

    with open(pred_file, "r") as f:
        pred_data = [json.loads(line) for line in f]  # {"idx":1, "label":0}

    with open(answer_file, "r") as f:
        answer_data = [json.loads(line) for line in f]  # {..., "idx":0, "label":false}

    correct_q = 0
    total_q = 0
    correct_ind = 0
    total_ind = 0
    assert len(pred_data) == len(answer_data), f"The prediction answer set ({len(pred_data)}) and answer answer set " \
                                               f"({len(answer_data)}) do not match in length"

    pred_labels = []
    #print(f"pred_data[0]: {pred_data[0]}")

    for line_dict in pred_data: # prediction is a dict:

        passage_questions = line_dict["passage"]["questions"] # this is a list of dicts with an "idx" and "answers" label

        for dict_answers in passage_questions:
            #print(f"dict_answers: {dict_answers}")
            question_labels = []
            for dict_labels in dict_answers["answers"]:
                #print(f"dict_labels: {dict_labels}")
                #pred_labels.append(dict_labels)
                question_labels.append(dict_labels)
            pred_labels.append(question_labels)

    print("pred_labels:", len(pred_labels))
    answer_labels = []
    for line_dict in answer_data:  # prediction is a dict:

        passage_questions = line_dict["passage"]["questions"]  # this is a list of dicts with an "idx" and "answers" label

        for dict_answers in passage_questions:
            # print(f"dict_answers: {dict_answers}")
            question_labels = []
            for dict_labels in dict_answers["answers"]:
                # print(f"dict_labels: {dict_labels}")
                # pred_labels.append(dict_labels)
                question_labels.append(dict_labels)
            answer_labels.append(question_labels)

    print(f"answer_labels: {len(answer_labels)}")

    assert len(pred_labels) == len(answer_labels), f"The lengths of pred_labels and answer_labels are not equal."

    for i in range(len(pred_labels)):
        assert len(pred_labels[i]) == len(answer_labels[i])
        is_correct = True
        for j in range(len(pred_labels[i])):
            if pred_labels[i][j]["idx"] == answer_labels[i][j]["idx"]:
                if pred_labels[i][j]["label"] != answer_labels[i][j]["label"]:
                    is_correct = False
                else: # they are equal
                    correct_ind += 1
            else: raise Exception(f"The idx values should be equal!")
            total_ind += 1
        if is_correct: correct_q += 1

        total_q += 1 # total is the number of questions, not answer options within.

    print(f"Total correct_q: {correct_q}\n"
          f"Total correct_ind {correct_ind}\n"
          f"Number of instances_q: {total_q}\n"
          f"Number of instances_ind: {total_ind}\n"
          f"Accuracy_q: {correct_q/total_q}\n"
          f"Accuracy_ind: {correct_ind/total_ind}")

    return correct_q, total_q, correct_ind, total_ind

def compare_two_jsonl_files_ACC(pred_file, answer_file): #CB_COPA_RTE_WiC_BoolQ_WSC

    with open(pred_file, "r") as f:
        pred_data = [json.loads(line) for line in f] # {"idx":1, "label":0}

    with open(answer_file, "r") as f:
        answer_data = [json.loads(line) for line in f] # {..., "idx":0, "label":false}

    correct = 0
    total = len(pred_data)
    total_check = 0
    assert len(pred_data) == len(answer_data), f"The prediction answer set ({len(pred_data)}) and answer answer set " \
                                               f"({len(answer_data)}) do not match in length"
    for prediction in pred_data: # prediction is a dict:

        # find a line with
        pred_idx = prediction["idx"]
        answer_dict = next((item for item in answer_data if int(item["idx"]) == int(pred_idx)), None)
        assert len([item for item in answer_data if int(item["idx"]) == int(pred_idx)]) == 1
        if answer_dict is None: raise Exception(f"The id {int(pred_idx)} has no associated answer index!")

        cor_ans = None
        pred = None
        if isinstance(answer_dict["label"], int) and not isinstance(answer_dict["label"], bool):
            cor_ans = answer_dict["label"]
            pred = prediction["label"]
        elif isinstance(answer_dict["label"], bool): # handles case for BoolQ and WiC where the label is provided in bool form.
            # our produced predicitons are in string form and is lower cased.
            cor_ans = "true" if answer_dict["label"] else "false"
            pred = prediction["label"].lower()
            #print(f"cor_ans: {cor_ans}\npred: {pred}")
        else: # elif string
            cor_ans = answer_dict["label"].lower()
            pred = prediction["label"].lower()
            #print(f"cor_ans: {cor_ans}\npred: {pred}")

        if cor_ans == pred:
            correct += 1
        total_check += 1
    assert total_check == total

    print(f"Total correct: {correct}\n"
          f"Number of instances: {total}\n"
          f"Accuracy: {correct/total}")

    return correct, total

def compare_two_jsonl_files_ACC_PIQA_SIQA(pred_file, answer_file): #CB_COPA_RTE_WiC_BoolQ_WSC

    with open(pred_file, "r") as f:
        pred_data = [json.loads(line) for line in f] # {"idx":1, "label":0}

    with open(answer_file, "r") as f:
        labels = f.readlines()

    correct = 0
    total = len(pred_data)
    total_check = 0
    assert len(pred_data) == len(labels), f"The prediction answer set ({len(pred_data)}) and answer answer set " \
                                               f"({len(answer_data)}) do not match in length"
    for prediction in pred_data: # prediction is a dict:

        # find a line with
        pred_idx = int(prediction["idx"]) # from 0 to N.
        cor_ans = int(labels[pred_idx])
        pred = int(prediction["label"])

        if cor_ans == pred:
            correct += 1
        total_check += 1
    assert total_check == total

    print(f"Total correct: {correct}\n"
          f"Number of instances: {total}\n"
          f"Accuracy: {correct/total}")

    return correct, total

def compare_two_jsonl_files_ACC_MED(pred_file, answer_file):

    with open(pred_file, "r") as f:
        pred_data = [json.loads(line) for line in f] # {"idx":1, "label":0}

    col_names = []
    data = []
    answer_data = []
    with open(answer_file, "r") as f:
        tsv_file = csv.reader(f, delimiter="\t")
        counter = 0
        for line in tsv_file:
            if counter != 0:
                data.append(line)
            else:
                col_names = line
            counter += 1
    idx_index = col_names.index("pairID")
    gold_label_index = col_names.index("gold_label")
    for example in data:
        answer_data.append({"idx":example[idx_index],"label":example[gold_label_index]})

    correct = 0
    total = len(pred_data)
    total_check = 0
    assert len(pred_data) == len(answer_data), f"The prediction answer set ({len(pred_data)}) and answer answer set " \
                                               f"({len(answer_data)}) do not match in length"
    for prediction in pred_data: # prediction is a dict:

        # find a line with
        pred_idx = prediction["idx"]
        answer_dict = next((item for item in answer_data if int(item["idx"]) == int(pred_idx)), None)
        assert len([item for item in answer_data if int(item["idx"]) == int(pred_idx)]) == 1
        if answer_dict is None: raise Exception(f"The id {int(pred_idx)} has no associated answer index!")

        cor_ans = None
        pred = None
        if isinstance(answer_dict["label"], int) and not isinstance(answer_dict["label"], bool):
            cor_ans = answer_dict["label"]
            pred = prediction["label"]
        elif isinstance(answer_dict["label"], bool): # handles case for BoolQ and WiC where the label is provided in bool form.
            # our produced predicitons are in string form and is lower cased.
            cor_ans = "true" if answer_dict["label"] else "false"
            pred = prediction["label"].lower()
            #print(f"cor_ans: {cor_ans}\npred: {pred}")
        else: # elif string
            cor_ans = "entailment" if answer_dict["label"] == "entailment" else "not_entailment"
            pred = prediction["label"].lower()
            #print(f"cor_ans: {cor_ans}\npred: {pred}")

        if cor_ans == pred:
            correct += 1
        total_check += 1
    assert total_check == total

    print(f"Total correct: {correct}\n"
          f"Number of instances: {total}\n"
          f"Accuracy: {correct/total}")

    return correct, total

def compare_two_jsonl_files_ACC_CQA(pred_file, answer_file): #CB_COPA_RTE_WiC_BoolQ_WSC

    with open(pred_file, "r") as f:
        pred_data = [json.loads(line) for line in f] # {"idx":1, "label":0}

    with open(answer_file, "r") as f:
        answer_data = [json.loads(line) for line in f] # {..., "idx":0, "label":false}

    correct = 0
    total = len(pred_data)
    total_check = 0
    assert len(pred_data) == len(answer_data), f"The prediction answer set ({len(pred_data)}) and answer answer set " \
                                               f"({len(answer_data)}) do not match in length"
    for prediction in pred_data: # prediction is a dict:

        # find a line with
        pred_idx = prediction["idx"]
        #print("pred_idx:", pred_idx[2:-1])
        #print(f"answer_data:", answer_data)
        answer_dict = next((item for item in answer_data if str(item["id"]) == str(pred_idx[2:-1])), None)
        assert len([item for item in answer_data if str(item["id"]) == str(pred_idx[2:-1])]) == 1, f"len of answer_dict: {len(answer_dict)}"
        if answer_dict is None: raise Exception(f"The id {str(pred_idx)} has no associated answer index!")

        cor_ans = None
        pred = None
        if isinstance(answer_dict["answerKey"], int) and not isinstance(answer_dict["answerKey"], bool):
            cor_ans = answer_dict["answerKey"]
            pred = prediction["label"]
        elif isinstance(answer_dict["answerKey"], bool): # handles case for BoolQ and WiC where the label is provided in bool form.
            # our produced predicitons are in string form and is lower cased.
            cor_ans = "true" if answer_dict["answerKey"] else "false"
            pred = prediction["label"].lower()
            #print(f"cor_ans: {cor_ans}\npred: {pred}")
        else: # elif string
            cor_ans = answer_dict["answerKey"].lower()
            pred = prediction["label"].lower()
            #print(f"cor_ans: {cor_ans}\npred: {pred}")

        if cor_ans == pred:
            correct += 1
        total_check += 1
    assert total_check == total

    print(f"Total correct: {correct}\n"
          f"Number of instances: {total}\n"
          f"Accuracy: {correct/total}")

    return correct, total

def compare_two_jsonl_files_ACC_SciTail(pred_file, answer_file): #SNLI

    with open(pred_file, "r") as f:
        pred_data = [json.loads(line) for line in f] # {"idx":1, "label":0}

    with open(answer_file, "r") as f:
        answer_data = [json.loads(line) for line in f] # {..., "idx":0, "label":false}

    correct = 0
    total = len(pred_data)
    total_check = 0
    assert len(pred_data) == len(answer_data), f"The prediction answer set ({len(pred_data)}) and answer answer set " \
                                               f"({len(answer_data)}) do not match in length"
    for prediction in pred_data: # prediction is a dict:

        # find a line with
        pred_idx = prediction["idx"]
        answer_dict = answer_data[pred_idx]
        #answer_dict = next((item for item in answer_data if int(item["idx"]) == int(pred_idx)), None)
        #assert len([item for item in answer_data if int(item["idx"]) == int(pred_idx)]) == 1
        #if answer_dict is None: raise Exception(f"The id {int(pred_idx)} has no associated answer index!")

        cor_ans = "entailment" if answer_dict["gold_label"] == "entails" else "not_entailment"
        pred = prediction["label"].lower()

        if cor_ans == pred:
            correct += 1
        total_check += 1
    assert total_check == total

    print(f"Total correct: {correct}\n"
          f"Number of instances: {total}\n"
          f"Accuracy: {correct/total}")

    return correct, total

def compare_two_jsonl_files_ACC_SNLI(pred_file, answer_file): #SNLI

    with open(pred_file, "r") as f:
        pred_data = [json.loads(line) for line in f] # {"idx":1, "label":0}

    with open(answer_file, "r") as f:
        answer_data = [json.loads(line) for line in f] # {..., "idx":0, "label":false}

    correct = 0
    total = len(pred_data)
    total_check = 0
    assert len(pred_data) == len(answer_data), f"The prediction answer set ({len(pred_data)}) and answer answer set " \
                                               f"({len(answer_data)}) do not match in length"
    for prediction in pred_data: # prediction is a dict:

        # find a line with
        pred_idx = prediction["idx"]
        answer_dict = answer_data[pred_idx]
        #answer_dict = next((item for item in answer_data if int(item["idx"]) == int(pred_idx)), None)
        #assert len([item for item in answer_data if int(item["idx"]) == int(pred_idx)]) == 1
        #if answer_dict is None: raise Exception(f"The id {int(pred_idx)} has no associated answer index!")

        cor_ans = None
        pred = None
        if isinstance(answer_dict["gold_label"], int) and not isinstance(answer_dict["gold_label"], bool):
            cor_ans = answer_dict["gold_label"]
            pred = prediction["label"]
        elif isinstance(answer_dict["gold_label"], bool): # handles case for BoolQ and WiC where the label is provided in bool form.
            # our produced predicitons are in string form and is lower cased.
            cor_ans = "true" if answer_dict["gold_label"] else "false"
            pred = prediction["label"].lower()
            #print(f"cor_ans: {cor_ans}\npred: {pred}")
        else: # elif string
            cor_ans = answer_dict["gold_label"].lower()
            pred = prediction["label"].lower()
            #print(f"cor_ans: {cor_ans}\npred: {pred}")

        if cor_ans == pred:
            correct += 1
        total_check += 1
    assert total_check == total

    print(f"Total correct: {correct}\n"
          f"Number of instances: {total}\n"
          f"Accuracy: {correct/total}")

    return correct, total

def compare_two_jsonl_files_ACC_MPE(pred_file, answer_file): #CB_COPA_RTE_WiC_BoolQ_WSC

    with open(pred_file, "r") as f:
        pred_data = [json.loads(line) for line in f] # {"idx":1, "label":0}

    #with open(answer_file, "r") as f:
    #    answer_data = [json.loads(line) for line in f] # {..., "idx":0, "label":false}
    dataLoader = MPELoader(filepath=answer_file, tokenizer=None, is_test=False, mode="softmax")
    answer_data = dataLoader.processed_data

    correct = 0
    total = len(pred_data)
    total_check = 0
    assert len(pred_data) == len(answer_data), f"The prediction answer set ({len(pred_data)}) and answer answer set " \
                                               f"({len(answer_data)}) do not match in length"
    for prediction in pred_data: # prediction is a dict:

        # find a line with
        pred_idx = prediction["idx"]
        answer_dict = next((item for item in answer_data if int(item["idx"]) == int(pred_idx)), None)
        assert len([item for item in answer_data if int(item["idx"]) == int(pred_idx)]) == 1, f"pred_idx: {pred_idx}"
        if answer_dict is None: raise Exception(f"The id {int(pred_idx)} has no associated answer index!")

        cor_ans = None
        pred = None
        print(f"answer_dict: {answer_dict['idx']} \t pred_idx: {pred_idx}")
        if isinstance(answer_dict["gold_label"], int) and not isinstance(answer_dict["gold_label"], bool):
            cor_ans = answer_dict["gold_label"]
            pred = prediction["label"]
        elif isinstance(answer_dict["gold_label"], bool): # handles case for BoolQ and WiC where the label is provided in bool form.
            # our produced predicitons are in string form and is lower cased.
            cor_ans = "true" if answer_dict["gold_label"] else "false"
            pred = prediction["label"].lower()
            #print(f"cor_ans: {cor_ans}\npred: {pred}")
        else: # elif string
            cor_ans = answer_dict["gold_label"].lower()
            pred = prediction["label"].lower()
            #print(f"cor_ans: {cor_ans}\npred: {pred}")

        if cor_ans == pred:
            correct += 1
        total_check += 1
    assert total_check == total

    print(f"Total correct: {correct}\n"
          f"Number of instances: {total}\n"
          f"Accuracy: {correct/total}")

    return correct, total

def compare_two_jsonl_files_ACC_RACE(pred_file, answer_file_middle, answer_file_high): #CB_COPA_RTE_WiC_BoolQ_WSC

    with open(pred_file, "r") as f:
        pred_data = [json.loads(line) for line in f] # {"idx":1, "label":0}

    #with open(answer_file, "r") as f:
    #    answer_data = [json.loads(line) for line in f] # {..., "idx":0, "label":false}
    dataLoader = RACELoader(filepath_middle=answer_file_middle, filepath_high=answer_file_high,
                            tokenizer=None, is_test=False, get_pred=True)
    answer_data = dataLoader.processed_data

    correct = 0
    total = len(pred_data)
    total_check = 0
    assert len(pred_data) == len(answer_data), f"The prediction answer set ({len(pred_data)}) and answer answer set " \
                                               f"({len(answer_data)}) do not match in length"
    for prediction in pred_data: # prediction is a dict:

        # find a line with
        pred_idx = prediction["idx"]
        #print("pred_idx:", pred_idx)
        answer_dict = next((item for item in answer_data if str(item["idx"]) == str(pred_idx[2:-1])), None)
        assert len([item for item in answer_data if str(item["idx"]) == str(pred_idx[2:-1])]) == 1, f"pred_idx: {pred_idx}"
        if answer_dict is None: raise Exception(f"The id {str(pred_idx[2:-1])} has no associated answer index!")

        cor_ans = None
        pred = None
        #print(f"answer_dict: {answer_dict['idx']} \t pred_idx: {pred_idx}")
        if isinstance(answer_dict["correct_label"], int) and not isinstance(answer_dict["correct_label"], bool):
            cor_ans = answer_dict["correct_label"]
            pred = prediction["label"]
        elif isinstance(answer_dict["correct_label"], bool): # handles case for BoolQ and WiC where the label is provided in bool form.
            # our produced predicitons are in string form and is lower cased.
            cor_ans = "true" if answer_dict["correct_label"] else "false"
            pred = prediction["label"].lower()
            #print(f"cor_ans: {cor_ans}\npred: {pred}")
        else: # elif string
            cor_ans = answer_dict["correct_label"].lower()
            pred = prediction["label"].lower()
            #print(f"cor_ans: {cor_ans}\npred: {pred}")

        if cor_ans == pred:
            correct += 1
        total_check += 1
    assert total_check == total

    print(f"Total correct: {correct}\n"
          f"Number of instances: {total}\n"
          f"Accuracy: {correct/total}")

    return correct, total

def compare_two_jsonl_files_ACC_ReClor(pred_file, answer_file): #CB_COPA_RTE_WiC_BoolQ_WSC

    with open(pred_file, "r") as f:
        pred_data = [json.loads(line) for line in f] # {"idx":1, "label":0}

    #with open(answer_file, "r") as f:
    #    answer_data = [json.loads(line) for line in f] # {..., "idx":0, "label":false}
    dataLoader = ReClorLoader(filepath=answer_file, tokenizer=None,
                              is_test=False, get_pred=True)
    answer_data = dataLoader.processed_data

    correct = 0
    total = len(pred_data)
    total_check = 0
    assert len(pred_data) == len(answer_data), f"The prediction answer set ({len(pred_data)}) and answer answer set " \
                                               f"({len(answer_data)}) do not match in length"
    for prediction in pred_data: # prediction is a dict:

        # find a line with
        pred_idx = prediction["idx"]
        #print("pred_idx:", pred_idx)
        answer_dict = next((item for item in answer_data if str(item["idx"]) == str(pred_idx[2:-1])), None)
        assert len([item for item in answer_data if str(item["idx"]) == str(pred_idx[2:-1])]) == 1, f"pred_idx: {pred_idx}"
        if answer_dict is None: raise Exception(f"The id {str(pred_idx[2:-1])} has no associated answer index!")

        cor_ans = None
        pred = None
        #print(f"answer_dict: {answer_dict['idx']} \t pred_idx: {pred_idx}")
        if isinstance(answer_dict["label"], int) and not isinstance(answer_dict["label"], bool):
            cor_ans = answer_dict["label"]
            pred = prediction["label"]
        elif isinstance(answer_dict["label"], bool): # handles case for BoolQ and WiC where the label is provided in bool form.
            # our produced predicitons are in string form and is lower cased.
            cor_ans = "true" if answer_dict["label"] else "false"
            pred = prediction["label"].lower()
            #print(f"cor_ans: {cor_ans}\npred: {pred}")
        else: # elif string
            cor_ans = answer_dict["label"].lower()
            pred = prediction["label"].lower()
            #print(f"cor_ans: {cor_ans}\npred: {pred}")

        if cor_ans == pred:
            correct += 1
        total_check += 1
    assert total_check == total

    print(f"Total correct: {correct}\n"
          f"Number of instances: {total}\n"
          f"Accuracy: {correct/total}")

    return correct, total

def compare_two_jsonl_files_ACC_DREAM(pred_file, answer_file): #CB_COPA_RTE_WiC_BoolQ_WSC

    with open(pred_file, "r") as f:
        pred_data = [json.loads(line) for line in f] # {"idx":1, "label":0}

    #with open(answer_file, "r") as f:
    #    answer_data = [json.loads(line) for line in f] # {..., "idx":0, "label":false}
    dataLoader = DREAMLoader(filepath=answer_file,
                             tokenizer=None, is_test=False, get_pred=True)
    answer_data = dataLoader.processed_data

    correct = 0
    total = len(pred_data)
    total_check = 0
    assert len(pred_data) == len(answer_data), f"The prediction answer set ({len(pred_data)}) and answer answer set " \
                                               f"({len(answer_data)}) do not match in length"
    for prediction in pred_data: # prediction is a dict:

        # find a line with
        pred_idx = prediction["idx"]
        answer_dict = next((item for item in answer_data if str(item["idx"]) == str(pred_idx[2:-1])), None)
        assert len([item for item in answer_data if str(item["idx"]) == str(pred_idx[2:-1])]) == 1, f"pred_idx: {pred_idx}"
        if answer_dict is None: raise Exception(f"The id {str(pred_idx[2:-1])} has no associated answer index!")

        cor_ans = None
        pred = None
        #print(f"answer_dict: {answer_dict['idx']} \t pred_idx: {pred_idx}")
        if isinstance(answer_dict["label"], int) and not isinstance(answer_dict["label"], bool):
            cor_ans = answer_dict["label"]
            pred = prediction["label"]
        elif isinstance(answer_dict["label"], bool):  # handles case for BoolQ and WiC where the label is provided in bool form.
            # our produced predicitons are in string form and is lower cased.
            cor_ans = "true" if answer_dict["label"] else "false"
            pred = prediction["label"].lower()
            # print(f"cor_ans: {cor_ans}\npred: {pred}")
        else:  # elif string
            cor_ans = answer_dict["label"].lower()
            pred = prediction["label"].lower()
            # print(f"cor_ans: {cor_ans}\npred: {pred}")

        if cor_ans == pred:
            correct += 1
        total_check += 1
    assert total_check == total

    print(f"Total correct: {correct}\n"
          f"Number of instances: {total}\n"
          f"Accuracy: {correct / total}")

    return correct, total

def compare_two_jsonl_files_ACC_StrategyQA(pred_file, answer_file): #CB_COPA_RTE_WiC_BoolQ_WSC

    with open(pred_file, "r") as f:
        pred_data = [json.loads(line) for line in f] # {"idx":1, "label":0}

    #with open(answer_file, "r") as f:
    #    answer_data = [json.loads(line) for line in f] # {..., "idx":0, "label":false}
    dataLoader = StrategyQALoader(filepath=answer_file,
                                  tokenizer=None, is_test=False)
    answer_data = dataLoader.processed_data

    correct = 0
    total = len(pred_data)
    total_check = 0
    assert len(pred_data) == len(answer_data), f"The prediction answer set ({len(pred_data)}) and answer answer set " \
                                               f"({len(answer_data)}) do not match in length"
    for prediction in pred_data: # prediction is a dict:

        # find a line with
        pred_idx = prediction["idx"]
        # item idx is in index 0; I've used list(zip(...)) here, hence, why its different.
        answer_dict = next((item for item in answer_data if str(item[0]) == str(pred_idx[2:-1])), None)
        assert len([item for item in answer_data if str(item[0]) == str(pred_idx[2:-1])]) == 1, f"pred_idx: {pred_idx}"
        if answer_dict is None: raise Exception(f"The id {str(pred_idx[2:-1])} has no associated answer index!")

        cor_ans = None
        pred = None
        #print(f"answer_dict: {answer_dict['idx']} \t pred_idx: {pred_idx}")
        if isinstance(answer_dict[3], int) and not isinstance(answer_dict[3], bool):
            cor_ans = answer_dict[3]
            pred = prediction["label"]
        elif isinstance(answer_dict[3], bool):  # handles case for BoolQ and WiC where the label is provided in bool form.
            # our produced predicitons are in string form and is lower cased.
            cor_ans = "true" if answer_dict[3] else "false"
            pred = prediction["label"].lower()
            # print(f"cor_ans: {cor_ans}\npred: {pred}")
        else:  # elif string
            cor_ans = answer_dict[3].lower()
            pred = prediction["label"].lower()
            # print(f"cor_ans: {cor_ans}\npred: {pred}")

        if cor_ans == pred:
            correct += 1
        total_check += 1
    assert total_check == total

    print(f"Total correct: {correct}\n"
          f"Number of instances: {total}\n"
          f"Accuracy: {correct / total}")

    return correct, total