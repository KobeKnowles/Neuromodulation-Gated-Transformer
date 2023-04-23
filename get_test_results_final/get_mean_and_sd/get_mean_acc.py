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

    dataset = "WiC"
    model_type = "no-gating-end-no-aux-toks"

    filename_exp1_pred_file_acc = "/data/kkno604/NGT_experiments_updated/general_experiments/" \
                                  "no-gating-end/exp1/Results/get_results_out_of_domain/StrategyQA/Accuracy/iteration200000.jsonl"
    filename_exp2_pred_file_acc = "/data/kkno604/NGT_experiments_updated/general_experiments/" \
                                  "no-gating-end/exp2/Results/get_results_out_of_domain/StrategyQA/Accuracy/iteration200000.jsonl"
    filename_exp3_pred_file_acc = "/data/kkno604/NGT_experiments_updated/general_experiments/" \
                                  "no-gating-end/exp3/Results/get_results_out_of_domain/StrategyQA/Accuracy/iteration200000.jsonl"

    #filename_exp1_pred_file_acc = "/data/kkno604/NGT_experiments_updated/aux_tok_ablation/no-gating-end-no-aux-toks" \
    #                              "/exp1/Results/results_WiC/Accuracy/iteration30000.jsonl"
    #filename_exp2_pred_file_acc = "/data/kkno604/NGT_experiments_updated/aux_tok_ablation/no-gating-end-no-aux-toks" \
    #                              "/exp2/Results/results_WiC/Accuracy/iteration30000.jsonl"
    #filename_exp3_pred_file_acc = "/data/kkno604/NGT_experiments_updated/aux_tok_ablation/no-gating-end-no-aux-toks" \
    #                              "/exp3/Results/results_WiC/Accuracy/iteration30000.jsonl"

    filename_list = [filename_exp1_pred_file_acc, filename_exp2_pred_file_acc, filename_exp3_pred_file_acc]
    acc_list = []
    for i, name in enumerate(filename_list):
        with open(name, "r") as f:
            lst = f.readlines()
            acc = lst[-1]
            assert "Accuracy:" in acc, f"acc: {acc}"
            acc = acc.replace("Accuracy:", "")
            acc_list.append(float(acc))


    for i in range(len(acc_list)): acc_list[i] = round(acc_list[i], 4)*100
    print(f"acc: {acc_list}")
    print(f"dataset: {dataset}")
    print(f"model_type: {model_type}")

    print(f"mean_acc: {mean(acc_list)}\n"
          f"standard_deviation_acc: {standard_deviation(acc_list)}")