from accuracy_functions import compare_two_jsonl_files_ACC_MPE
from f1_score_functions import compare_two_jsonl_files_f1_unweighted_MPE
from save_functions import save_results, save_results_CB_f1
import json

if __name__ == "__main__":

    for exp in range(1,4):
        for iteration in [50000, 100000, 150000, 200000]:
            for type_ in ["gating-end","no-gating-end","no-gating-no-extra-layers"]:

                pred_file = "/data/kkno604/NGT_experiments_updated/general_experiments/"+type_+"/exp"+str(exp)+"/Results/" \
                            "get_results_in_domain/MPE/prediction_files/iteration"+str(iteration)+".jsonl"

                answer_file = "/large_data/MPE/MultiPremiseEntailment-master/data/MPE/mpe_dev.txt"

                save_filepath = "/data/kkno604/NGT_experiments_updated/general_experiments/"+type_+"/exp"+str(exp)+"/Results/" \
                                "get_results_in_domain/MPE/Accuracy/iteration"+str(iteration)+".jsonl"
                save_filepathf1 = "/data/kkno604/NGT_experiments_updated/general_experiments/"+type_+"/" \
                                  "exp"+str(exp)+"/Results/get_results_in_domain/MPE/F1/iteration"+str(iteration)+".jsonl"

                correct, total = compare_two_jsonl_files_ACC_MPE(pred_file=pred_file, answer_file=answer_file)
                entail_f1_score, neutral_f1_score, contradiction_f1_score, f1_score = \
                    compare_two_jsonl_files_f1_unweighted_MPE(pred_file=pred_file, answer_file=answer_file)

                save_results(save_filepath=save_filepath, correct=correct, total=total, i=iteration)
                save_results_CB_f1(save_filepath=save_filepathf1, entail_f1_score=entail_f1_score,
                                   neutral_f1_score=neutral_f1_score, contradiction_f1_score=contradiction_f1_score,
                                   f1_score=f1_score, i=iteration)