import math
import statistics

def mean(lst):
    return round(sum(lst)/len(lst),2)

def median(lst):
    lst.sort()
    return round(statistics.median(lst),2)

def standard_deviation(lst):
    return round(statistics.stdev(lst),2)

if __name__ == "__main__":

    dataset = "ReCoRD"
    model_type = "no-gating-no-extra-layers"

    # here the f1 file holds both the exact-match accuracy and f1 score.
    filename_exp1_pred_file_f1 = "/data/kkno604/NGT_experiments_updated/general_experiments/" \
                                  "no-gating-no-extra-layers/exp1/Results/get_results_in_domain/ReCoRD/F1/iteration200000.jsonl"
    filename_exp2_pred_file_f1 = "/data/kkno604/NGT_experiments_updated/general_experiments/" \
                                  "no-gating-no-extra-layers/exp2/Results/get_results_in_domain/ReCoRD/F1/iteration200000.jsonl"
    filename_exp3_pred_file_f1 = "/data/kkno604/NGT_experiments_updated/general_experiments/" \
                                  "no-gating-no-extra-layers/exp3/Results/get_results_in_domain/ReCoRD/F1/iteration200000.jsonl"

    filename_list = [filename_exp1_pred_file_f1, filename_exp2_pred_file_f1, filename_exp3_pred_file_f1]
    acc_list, f1_list = [], []
    for i, name in enumerate(filename_list):
        with open(name, "r") as f:
            lst = f.readlines()
            acc = lst[-1]
            f1 = lst[-2]
            assert "exact-match:" in acc, f"acc: {acc}"
            acc = acc.replace("exact-match:", "")
            acc_list.append(float(acc))
            assert "f1-score:" in f1, f"f1-score: {f1}"
            f1 = f1.replace("f1-score:", "")
            f1_list.append(float(f1))

    print(f"acc: {acc_list}\nf1: {f1_list}")
    assert len(acc_list) == len(f1_list)

    #for i in range(len(acc_list)):
    #    acc_list[i] = round(acc_list[i], 4) * 100
    #    f1_list[i] = round(f1_list[i], 4) * 100  # *100 to convert to percentages.

    print(f"dataset: {dataset}")
    print(f"model_type: {model_type}")

    print(f"mean_acc: {mean(acc_list)}\n"
          f"standard_deviation_acc: {standard_deviation(acc_list)}\n"
          f"mean_f1: {mean(f1_list)}\n"
          f"standard_deviation_f1: {standard_deviation(f1_list)}")