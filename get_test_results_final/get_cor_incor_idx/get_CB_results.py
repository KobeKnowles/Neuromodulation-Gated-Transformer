import sys
sys.path.append("../..")

from get_test_results_final.get_cor_incor_idx.accuracy_functions import compare_two_jsonl_files_ACC

def CB_return_correct_incorrect_ids(pred_file, answer_file="/large_data/SuperGlue/CB/val.jsonl"):

    correct_idx_list, incorrect_idx_list = compare_two_jsonl_files_ACC(pred_file=pred_file, answer_file=answer_file)
    return correct_idx_list, incorrect_idx_list


'''
if __name__ == "__main__":

    for exp in range(1,4):
        for iteration in [50000, 100000, 150000, 200000]:
            for type_ in ["gating-end","no-gating-end","no-gating-no-extra-layers"]:
    #for exp in range(1, 4):
    #    for iteration in [10000, 20000, 30000]:
    #        for type_ in ["gating-end-aux-toks", "no-gating-end-aux-toks",
    #                      "gating-end-no-aux-toks", "no-gating-end-no-aux-toks"]:

                pred_file = "/data/kkno604/NGT_experiments_updated/general_experiments/"+type_+"/exp"+str(exp)+"/Results/" \
                            "get_results_in_domain/CB/prediction_files/iteration"+str(iteration)+".jsonl"

                #pred_file = "/data/kkno604/NGT_experiments_updated/aux_tok_ablation/"+type_+"/" \
                #            "exp"+str(exp)+"/Results/results_CB/prediction_files/iteration"+str(iteration)+".jsonl"

                answer_file = "/large_data/SuperGlue/CB/val.jsonl"

                save_filepath = "/data/kkno604/NGT_experiments_updated/general_experiments/"+type_+"/exp"+str(exp)+"/Results/" \
                                "get_results_in_domain/CB/Accuracy/iteration"+str(iteration)+".jsonl"
                save_filepathf1 = "/data/kkno604/NGT_experiments_updated/general_experiments/"+type_+"/" \
                                  "exp"+str(exp)+"/Results/get_results_in_domain/CB/F1/iteration"+str(iteration)+".jsonl"

                #save_filepath = "/data/kkno604/NGT_experiments_updated/aux_tok_ablation/"+type_+"/" \
                #                    "exp"+str(exp)+"/Results/results_CB/Accuracy/iteration"+str(iteration)+".jsonl"
                #save_filepathf1 = "/data/kkno604/NGT_experiments_updated/aux_tok_ablation/"+type_+"/" \
                #                    "exp"+str(exp)+"/Results/results_CB/F1/iteration"+str(iteration)+".jsonl"

                correct, total = compare_two_jsonl_files_ACC(pred_file=pred_file, answer_file=answer_file)
                entail_f1_score, neutral_f1_score, contradiction_f1_score, f1_score = \
                    compare_two_jsonl_files_f1_unweighted_CB(pred_file=pred_file, answer_file=answer_file)

                save_results(save_filepath=save_filepath, correct=correct, total=total, i=iteration)
                save_results_CB_f1(save_filepath=save_filepathf1, entail_f1_score=entail_f1_score,
                                   neutral_f1_score=neutral_f1_score, contradiction_f1_score=contradiction_f1_score,
                                   f1_score=f1_score, i=iteration)
'''