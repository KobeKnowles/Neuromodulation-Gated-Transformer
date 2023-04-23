from accuracy_functions import compare_two_jsonl_files_ACC_DREAM
from save_functions import save_results
import json

if __name__ == "__main__":

    for exp in range(1,4):
        for iteration in [50000, 100000, 150000, 200000]:
            for type_ in ["gating-end", "no-gating-end", "no-gating-no-extra-layers"]:
                pred_file = "/data/kkno604/NGT_experiments_updated/general_experiments/"+type_+"/exp"+str(exp)+"/Results/" \
                            "get_results_out_of_domain/DREAM/prediction_files/iteration"+str(iteration)+".jsonl"

                answer_file = "/large_data/DREAM/dream/data/dev.json"

                save_filepath = "/data/kkno604/NGT_experiments_updated/general_experiments/"+type_+"/exp"+str(exp)+"/Results/" \
                                "get_results_out_of_domain/DREAM/Accuracy/iteration"+str(iteration)+".jsonl"

                correct, total = compare_two_jsonl_files_ACC_DREAM(pred_file=pred_file, answer_file=answer_file)

                save_results(save_filepath=save_filepath, correct=correct, total=total, i=iteration)