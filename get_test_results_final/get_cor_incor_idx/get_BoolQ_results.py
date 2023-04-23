import sys
sys.path.append("../..")

from get_test_results_final.get_cor_incor_idx.accuracy_functions import compare_two_jsonl_files_ACC


def BoolQ_return_correct_incorrect_ids(pred_file, answer_file="/large_data/SuperGlue/BoolQ/val.jsonl"):

    correct_idx_list, incorrect_idx_list = compare_two_jsonl_files_ACC(pred_file=pred_file, answer_file=answer_file)
    return correct_idx_list, incorrect_idx_list


'''
if __name__ == "__main__":

    for exp in range(1,4):
        for iteration in [10000, 20000, 30000]:
            for type_ in ["gating-end-aux-toks", "no-gating-end-aux-toks",
                          "gating-end-no-aux-toks", "no-gating-end-no-aux-toks"]:
                pred_file = "/data/kkno604/NGT_experiments_updated/aux_tok_ablation/"+type_+"/" \
                            "exp"+str(exp)+"/Results/results_BoolQ/prediction_files/iteration"+str(iteration)+".jsonl"

                answer_file = "/large_data/SuperGlue/BoolQ/val.jsonl"
                save_filepath = "/data/kkno604/NGT_experiments_updated/aux_tok_ablation/"+type_+"/" \
                                "exp"+str(exp)+"/Results/results_BoolQ/Accuracy/iteration"+str(iteration)+".jsonl"

                correct, total = compare_two_jsonl_files_ACC(pred_file=pred_file, answer_file=answer_file)

                save_results(save_filepath=save_filepath, correct=correct, total=total, i=iteration)
'''