import sys
sys.path.append("../..")

from get_test_results_final.get_cor_incor_idx.accuracy_functions import compare_two_jsonl_files_ACC_RACE


def RACE_return_correct_incorrect_ids(pred_file, answer_file_middle="/large_data/RACE/RACE/dev/middle/",
                                         answer_file_high="/large_data/RACE/RACE/dev/high/"):

    correct_idx_list, incorrect_idx_list = compare_two_jsonl_files_ACC_RACE(pred_file=pred_file,
                                                                  answer_file_middle=answer_file_middle,
                                                                  answer_file_high=answer_file_high)
    return correct_idx_list, incorrect_idx_list

'''
if __name__ == "__main__":

    for exp in range(1, 4):
        for iteration in [50000, 100000, 150000, 200000]:
            for type_ in ["gating-end", "no-gating-end", "no-gating-no-extra-layers"]:
                pred_file = "/data/kkno604/NGT_experiments_updated/general_experiments/"+type_+"/exp"+str(exp)+"/Results/" \
                            "get_results_in_domain/RACE/prediction_files/iteration"+str(iteration)+".jsonl"

                answer_file_middle = "/large_data/RACE/RACE/dev/middle/"
                answer_file_high = "/large_data/RACE/RACE/dev/high/"

                save_filepath = "/data/kkno604/NGT_experiments_updated/general_experiments/"+type_+"/exp"+str(exp)+"/Results/" \
                                "get_results_in_domain/RACE/Accuracy/iteration"+str(iteration)+".jsonl"

                correct, total = compare_two_jsonl_files_ACC_RACE(pred_file=pred_file,
                                                                  answer_file_middle=answer_file_middle,
                                                                  answer_file_high=answer_file_high)

                save_results(save_filepath=save_filepath, correct=correct, total=total, i=iteration)
'''