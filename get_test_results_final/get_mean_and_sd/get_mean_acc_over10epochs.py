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

    dataset = "WSC"
    model_type = "gating-start"

    acc_list = [0.0, 0.0, 0.0]  # hardcoded for three experiments.
    for epoch in range(1,4):
        #filename_exp1_pred_file_acc = "/data/kkno604/NGT_experiments_updated/superGLUE-experiments/WSC/bert-large/" \
        #                              "no-gating-start/exp1/Results/val-self-reported-accuracy/epoch"+str(epoch)+".txt"
        #filename_exp2_pred_file_acc = "/data/kkno604/NGT_experiments_updated/superGLUE-experiments/WSC/bert-large/" \
        #                              "no-gating-start/exp2/Results/val-self-reported-accuracy/epoch"+str(epoch)+".txt"
        #filename_exp3_pred_file_acc = "/data/kkno604/NGT_experiments_updated/superGLUE-experiments/WSC/bert-large/" \
        #                              "no-gating-start/exp3/Results/val-self-reported-accuracy/epoch"+str(epoch)+".txt"

        filename_exp1_pred_file_acc = "/data/kkno604/NGT_experiments_updated/hyperparameter-search/WSC/BERT-large/" \
                                      "gating-start/exp1/Results/val-self-reported-metric/epoch"+str(epoch)+".txt"
        filename_exp2_pred_file_acc = "/data/kkno604/NGT_experiments_updated/hyperparameter-search/WSC/BERT-large/" \
                                      "gating-start/exp2/Results/val-self-reported-metric/epoch"+str(epoch)+".txt"
        filename_exp3_pred_file_acc = "/data/kkno604/NGT_experiments_updated/hyperparameter-search/WSC/BERT-large/" \
                                      "gating-start/exp3/Results/val-self-reported-metric/epoch"+str(epoch) + ".txt"

        filename_list = [filename_exp1_pred_file_acc, filename_exp2_pred_file_acc, filename_exp3_pred_file_acc]
        for i, name in enumerate(filename_list):
            with open(name, "r") as f:
                lst = f.readlines()
                acc = lst[-1]
                assert "Accuracy:" in acc, f"acc: {acc}"
                acc = acc.replace("Accuracy:", "")
                acc_list[i] = max(float(acc_list[i]), float(acc))

    print(f"acc: {acc_list}")
    for i in range(len(acc_list)): acc_list[i] = round(acc_list[i], 4)*100
    print(f"acc: {acc_list}")
    print(f"dataset: {dataset}")
    print(f"model_type: {model_type}")

    print(f"mean_acc: {mean(acc_list)}\n"
          f"standard_deviation_acc: {standard_deviation(acc_list)}")