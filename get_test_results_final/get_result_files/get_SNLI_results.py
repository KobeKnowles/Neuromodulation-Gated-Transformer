from accuracy_functions import compare_two_jsonl_files_ACC_SNLI
from f1_score_functions import compare_two_jsonl_files_f1_unweighted_SNLI
from save_functions import save_results, save_results_CB_f1
import json

if __name__ == "__main__":

    #iteration = 200000
    #exp = 3
    for exp in range(1,4):
        for iteration in [50000, 100000, 150000, 200000]:
            for type_ in ["gating-end","no-gating-end","no-gating-no-extra-layers"]:
                #type_ = "gating-end"
                pred_file = "/data/kkno604/NGT_experiments_updated/general_experiments/"+type_+"/exp"+str(exp)+"/Results/" \
                            "get_results_out_of_domain/SNLI/prediction_files/iteration"+str(iteration)+".jsonl"

                answer_file = "/large_data/SNLI_1.0/snli_1.0/snli_1.0_dev.jsonl"

                save_filepath = "/data/kkno604/NGT_experiments_updated/general_experiments/"+type_+"/exp"+str(exp)+"/Results/" \
                                "get_results_out_of_domain/SNLI/Accuracy/iteration"+str(iteration)+".jsonl"
                save_filepathf1 = "/data/kkno604/NGT_experiments_updated/general_experiments/"+type_+"/" \
                                  "exp"+str(exp)+"/Results/get_results_out_of_domain/SNLI/F1/iteration"+str(iteration)+".jsonl"

                correct, total = compare_two_jsonl_files_ACC_SNLI(pred_file=pred_file, answer_file=answer_file)
                entail_f1_score, neutral_f1_score, contradiction_f1_score, f1_score = \
                    compare_two_jsonl_files_f1_unweighted_SNLI(pred_file=pred_file, answer_file=answer_file)

                save_results(save_filepath=save_filepath, correct=correct, total=total, i=iteration)
                save_results_CB_f1(save_filepath=save_filepathf1, entail_f1_score=entail_f1_score,
                                   neutral_f1_score=neutral_f1_score, contradiction_f1_score=contradiction_f1_score,
                                   f1_score=f1_score, i=iteration)