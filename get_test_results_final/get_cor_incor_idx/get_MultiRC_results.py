import sys
sys.path.append("../..")

from get_test_results_final.get_cor_incor_idx.accuracy_functions import compare_two_jsonl_files_MultiRC_ACC


def MultiRC_return_correct_incorrect_ids(pred_file, answer_file="/large_data/SuperGlue/MultiRC/val.jsonl"):

    correct_idx_list, incorrect_idx_list = compare_two_jsonl_files_MultiRC_ACC(pred_file=pred_file, answer_file=answer_file)
    return correct_idx_list, incorrect_idx_list

'''
if __name__ == "__main__":

    dataset = "MultiRC"
    for exp in range(1,4):
        for iteration in [10000, 20000, 30000]:
            for type_ in ["gating-end-aux-toks", "no-gating-end-aux-toks",
                          "gating-end-no-aux-toks", "no-gating-end-no-aux-toks"]:
                pred_file = "/data/kkno604/NGT_experiments_updated/general_experiments/"+type_+"/exp"+str(exp)+"/Results/" \
                            "get_results_in_domain/"+dataset+"/prediction_files/iteration"+str(iteration)+".jsonl"

                answer_file = "/large_data/SuperGlue/"+dataset+"/val.jsonl"

                save_filepath = "/data/kkno604/NGT_experiments_updated/general_experiments/"+type_+"/exp"+str(exp)+"/Results/" \
                                "get_results_in_domain/"+dataset+"/EM/iteration"+str(iteration)+".jsonl"
                save_filepathf1 = "/data/kkno604/NGT_experiments_updated/general_experiments/"+type_+"/exp"+str(exp)+"/Results/" \
                                    "get_results_in_domain/"+dataset+"/F1/iteration"+str(iteration)+".jsonl"

                correct_q, total_q, correct_ind, total_ind, = compare_two_jsonl_files_MultiRC_ACC(pred_file=pred_file, answer_file=answer_file)
                f1a_socre = compare_two_jsonl_files_MultiRC_F1a_2(pred_file=pred_file, answer_file=answer_file)

                save_results_MultiRC_ACC(save_filepath=save_filepath, correct_q=correct_q, total_q=total_q, correct_ind=correct_ind,
                                         total_ind=total_ind, i=iteration)
                save_results_MultiRC_f1a(save_filepath=save_filepathf1, f1a_score=f1a_socre, i=iteration)
'''