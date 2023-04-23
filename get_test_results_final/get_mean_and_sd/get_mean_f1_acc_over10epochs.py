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

    dataset = "MultiRC"
    model_type = "gating-end"

    acc_list, f1_list = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    f1_acc_pair_max = [0.0, 0.0, 0.0]
    f1_acc_pair = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]

    for epoch in range(1,4):
        #filename_exp1_pred_file_acc = "/data/kkno604/NGT_experiments_updated/superGLUE-experiments/MultiRC/bert-large/" \
        #                              "no-gating--start/exp1/Results/val-self-reported-accuracy/epoch"+str(epoch)+".txt"
        #filename_exp1_pred_file_f1 = "/data/kkno604/NGT_experiments_updated/superGLUE-experiments/MultiRC/bert-large/" \
        #                             "no-gating--start/exp1/Results/val-self-reported-macro-f1-score/epoch"+str(epoch)+".txt"
        #filename_exp2_pred_file_acc = "/data/kkno604/NGT_experiments_updated/superGLUE-experiments/MultiRC/bert-large/" \
        #                              "no-gating--start/exp2/Results/val-self-reported-accuracy/epoch"+str(epoch)+".txt"
        #filename_exp2_pred_file_f1 = "/data/kkno604/NGT_experiments_updated/superGLUE-experiments/MultiRC/bert-large/" \
        #                             "no-gating--start/exp2/Results/val-self-reported-macro-f1-score/epoch"+str(epoch)+".txt"
        #filename_exp3_pred_file_acc = "/data/kkno604/NGT_experiments_updated/superGLUE-experiments/MultiRC/bert-large/" \
        #                              "no-gating--start/exp3/Results/val-self-reported-accuracy/epoch"+str(epoch)+".txt"
        #filename_exp3_pred_file_f1 = "/data/kkno604/NGT_experiments_updated/superGLUE-experiments/MultiRC/bert-large/" \
        #                             "no-gating--start/exp3/Results/val-self-reported-macro-f1-score/epoch"+str(epoch)+".txt"

        filename_exp1_pred_file_acc = "/data/kkno604/NGT_experiments_updated/hyperparameter-search/MultiRC_2/BERT-large/" \
                                      "gating-end/exp1/Results/val-self-reported-metric/epoch"+str(epoch)+".txt"
        filename_exp1_pred_file_f1 = "/data/kkno604/NGT_experiments_updated/hyperparameter-search/MultiRC_2/BERT-large/" \
                                     "gating-end/exp1/Results/val-self-reported-macro-f1-score/epoch"+str(epoch)+".txt"
        filename_exp2_pred_file_acc = "/data/kkno604/NGT_experiments_updated/hyperparameter-search/MultiRC_2/BERT-large/" \
                                      "gating-end/exp2/Results/val-self-reported-metric/epoch"+str(epoch)+".txt"
        filename_exp2_pred_file_f1 = "/data/kkno604/NGT_experiments_updated/hyperparameter-search/MultiRC_2/BERT-large/" \
                                     "gating-end/exp2/Results/val-self-reported-macro-f1-score/epoch"+str(epoch)+".txt"
        filename_exp3_pred_file_acc = "/data/kkno604/NGT_experiments_updated/hyperparameter-search/MultiRC_2/BERT-large/" \
                                      "gating-end/exp3/Results/val-self-reported-metric/epoch"+str(epoch)+".txt"
        filename_exp3_pred_file_f1 = "/data/kkno604/NGT_experiments_updated/hyperparameter-search/MultiRC_2/BERT-large/" \
                                     "gating-end/exp3/Results/val-self-reported-macro-f1-score/epoch"+str(epoch)+".txt"

        filename_list = [filename_exp1_pred_file_f1, filename_exp1_pred_file_acc,
                         filename_exp2_pred_file_f1, filename_exp2_pred_file_acc,
                         filename_exp3_pred_file_f1, filename_exp3_pred_file_acc]

        for i, name in enumerate(filename_list):
            with open(name, "r") as f:
                lst = f.readlines()
                if i % 2 == 1: # odd
                    if dataset == "MultiRC":
                        acc = lst[-2]
                        assert "Accuracy_q:" in acc, f"acc: {acc}"
                        acc = acc.replace("Accuracy_q:", "")
                        f1_acc_pair[i//2][1] = float(acc)
                        if statistics.mean(f1_acc_pair[i//2]) > f1_acc_pair_max[i//2]:
                            f1_acc_pair_max[i//2] = statistics.mean(f1_acc_pair[i//2])
                            acc_list[i//2] = f1_acc_pair[i//2][1]
                            f1_list[i//2] = f1_acc_pair[i//2][0]
                    else:
                        acc = lst[-1]
                        assert "Accuracy:" in acc, f"acc: {acc}"
                        acc = acc.replace("Accuracy:", "")
                        f1_acc_pair[i//2][1] = float(acc)
                        if statistics.mean(f1_acc_pair[i//2]) > f1_acc_pair_max[i//2]:
                            f1_acc_pair_max[i//2] = statistics.mean(f1_acc_pair[i//2])
                            acc_list[i//2] = f1_acc_pair[i//2][1]
                            f1_list[i//2] = f1_acc_pair[i//2][0]
                else: # even - including 0
                    if dataset == "MultiRC":
                        f1 = lst[-1]
                        assert "f1a_score:" in f1, f"f1: {f1}"
                        f1 = f1.replace("f1a_score:", "")
                        #f1_list.append(float(f1))
                        f1_acc_pair[i // 2][0] = float(f1)
                        #f1_list[i//2] = max(f1_list[i//2], float(f1))
                    else:
                        f1 = lst[-1]
                        assert "f1_score_macro:" in f1, f"f1: {f1}"
                        f1 = f1.replace("f1_score_macro:", "")
                        #f1_list.append(float(f1))
                        f1_acc_pair[i//2][0] = float(f1)
                        #f1_list[i // 2] = max(f1_list[i // 2], float(f1))

    print(f"acc: {acc_list}\nf1: {f1_list}")
    assert len(acc_list) == len(f1_list)

    for i in range(len(acc_list)):
        acc_list[i] = round(acc_list[i], 4)*100
        f1_list[i] = round(f1_list[i], 4)*100 # *100 to convert to percentages.

    print(f"dataset: {dataset}")
    print(f"model_type: {model_type}")

    print(f"mean_acc: {mean(acc_list)}\n"
          f"standard_deviation_acc: {standard_deviation(acc_list)}\n"
          f"mean_f1: {mean(f1_list)}\n"
          f"standard_deviation_f1: {standard_deviation(f1_list)}")