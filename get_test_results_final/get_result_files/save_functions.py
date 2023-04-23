def save_results(save_filepath, correct, total, i):

    with open(save_filepath, "w") as f:
        f.write(f"epoch: {i}\n"
                f"Correct: {correct}\n"
                f"Total: {total}\n"
                f"Accuracy: {correct/total}")

def save_results_MultiRC_ACC(save_filepath, correct_q, total_q, correct_ind, total_ind, i):

    with open(save_filepath, "w") as f:
        f.write(f"epoch: {i}\n"
                f"Correct_q: {correct_q}\n"
                f"Total_q: {total_q}\n"
                f"Correct_ind: {correct_ind}\n"
                f"Total_ind: {total_ind}\n"
                f"Accuracy_q: {correct_q/total_q}\n"
                f"Accuracy_ind: {correct_ind/total_ind}")

def save_results_CB_f1(save_filepath, entail_f1_score, neutral_f1_score, contradiction_f1_score, f1_score, i):

    with open(save_filepath, "w") as f:
        f.write(f"epoch: {i}\n"
                f"entail_f1_score: {entail_f1_score}\n"
                f"neutral_f1_score: {neutral_f1_score}\n"
                f"contradiction_f1_score: {contradiction_f1_score}\n"
                f"f1_score_macro: {f1_score}\n")

def save_results_MultiRC_f1a(save_filepath, f1a_score, i):

    with open(save_filepath, "w") as f:
        f.write(f"epoch: {i}\n"
                f"f1a_score: {f1a_score}\n")