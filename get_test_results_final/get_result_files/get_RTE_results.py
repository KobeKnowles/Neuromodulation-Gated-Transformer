from accuracy_functions import compare_two_jsonl_files_ACC
from save_functions import save_results
import json

if __name__ == "__main__":

    #iteration = 200000
    #exp = 3
    #type_ = "no-gating-no-extra-layers"
    dataset = "RTE"
    for exp in range(1,4):
        for iteration in [10000, 20000, 30000]:
            for type_ in ["gating-end-aux-toks", "no-gating-end-aux-toks",
                          "gating-end-no-aux-toks", "no-gating-end-no-aux-toks"]:
                pred_file = "/data/kkno604/NGT_experiments_updated/general_experiments/"+type_+"/exp"+str(exp)+"/Results/" \
                            "get_results_in_domain/"+dataset+"/prediction_files/iteration"+str(iteration)+".jsonl"

                answer_file = "/large_data/SuperGlue/"+dataset+"/val.jsonl"

                save_filepath = "/data/kkno604/NGT_experiments_updated/general_experiments/"+type_+"/exp"+str(exp)+"/Results/" \
                                "get_results_in_domain/"+dataset+"/Accuracy/iteration"+str(iteration)+".jsonl"


                correct, total = compare_two_jsonl_files_ACC(pred_file=pred_file, answer_file=answer_file)
                save_results(save_filepath=save_filepath, correct=correct, total=total, i=20000)