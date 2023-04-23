from accuracy_functions import compare_two_jsonl_files_ACC_SciTail
from f1_score_functions import compare_two_jsonl_files_f1_unweighted_CB
from save_functions import save_results, save_results_CB_f1
import json

if __name__ == "__main__":

    #iteration = 200000
    #exp = 3
    for exp in range(1,4):
        for iteration in [50000, 100000, 150000, 200000]:
            for type_ in ["gating-end", "no-gating-end", "no-gating-no-extra-layers"]:
                #type_ = "no-gating-no-extra-layers"
                pred_file = "/data/kkno604/NGT_experiments_updated/general_experiments/"+type_+"/exp"+str(exp)+"/Results/" \
                            "get_results_in_domain/SciTail/prediction_files/iteration"+str(iteration)+".jsonl"

                answer_file = "/large_data/SciTail/SciTailV1.1/predictor_format/scitail_1.0_structure_dev.jsonl"

                save_filepath = "/data/kkno604/NGT_experiments_updated/general_experiments/"+type_+"/exp"+str(exp)+"/Results/" \
                                "get_results_in_domain/SciTail/Accuracy/iteration"+str(iteration)+".jsonl"

                correct, total = compare_two_jsonl_files_ACC_SciTail(pred_file=pred_file, answer_file=answer_file)

                save_results(save_filepath=save_filepath, correct=correct, total=total, i=iteration)