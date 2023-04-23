import re
import json

def fix_answer(answer, idx):

    # double check that all of below don't change something by accident.
    answer_old = answer
    answer = answer.replace("U. S. A.", "U.S.A.")
    answer = answer.replace("U. S.","U.S.")
    answer = answer.replace(" - ", "-")
    answer = answer.replace("U. N. ", "U.N.")
    answer = answer.replace(". org", ".org")
    answer = answer.replace(". com", ".com")
    answer = answer.replace(". fm", ".fm")
    #answer = answer.replace("D. C.", "D.C.")
    answer = answer.replace("D. C", "D.C") # this will change the above line as well. # be careful; note that sentences that end with a d. and start with a c in the next sentence will be brought together, but this may not impact answers. Double check to be sure.
    answer = answer.replace("M. D.", "M.D.")
    answer = answer.replace("J. R.", "J.R.")
    answer = answer.replace("S. S.", "S.S.")
    answer = answer.replace("Jane Doe No. 2", "Jane Doe No.2")
    if idx == 7367: # there is another idx where the space is actually correct. Hence, why the exact idx here.
        answer = answer.replace("E. coli", "E.coli")

    # added in gating-start experiments - they don't impact gating-end. 1st experiment.
    answer = answer.replace("U. K.", "U.K.")
    answer = answer.replace("World no. 1 McIlroy", "World no.1 McIlroy")
    answer = answer.replace("jennifer. newton @ mailonline. co. uk", "jennifer.newton@mailonline.co.uk")
    answer = answer.replace("Amazon. co. uk", "Amazon.co.uk")
    # added in gating-start experiment 3.
    answer = answer.replace("T. I.", "T.I.")
    answer = answer.replace("No I. D.", "No I.D.")
    answer = answer.replace("L. A. NAACP", "L.A. NAACP")
    answer = answer.replace("S. C.", "S.C.")
    answer = answer.replace("No. 189 Marc Warren", "No.189 Marc Warren")
    answer = answer.replace(".357 Magnum", ".357 Magnum")
    # added in no-gating-end experiment 1.
    if idx == 7776:
        answer = answer.replace("L. K. Bennett", "L.K.Bennett")
    # added in no-gating-end experiment 2
    answer = answer.replace("pets. In", "pets.In") # this is actuall correct in all instances in the validation file.
    answer = answer.replace("Ph. D","Ph.D.")
    answer = answer.replace(". tv", ".tv")
    answer = answer.replace("Domain. com. au", "Domain.com.au")
    answer = answer.replace("KAYAK. com. au", "KAYAK.com.au") # saw once instance "argument. The"; it is correct as is.
    answer = answer.replace("H. M. Prasetyo", "H.M. Prasetyo")
    answer = answer.replace("P. O.", "P.O.")
    answer = answer.replace("L. A.", "L.A.")
    answer = answer.replace("N. C. State", "N.C. State")
    answer = answer.replace("B. C.", "B.C.")
    answer = answer.replace("WomenFreebies. co. uk", "WomenFreebies.co.uk")
    answer = answer.replace("J. J.", "J.J.")
    answer = answer.replace("Ronald \" R. J. \" Williams Jr.", "Ronald \" R.J. \" Williams Jr.")
    answer = answer.replace("N. H.", "N.H.")
    answer = answer.replace("E. L. James", "E.L. James")
    # added in no-gating-end experiment 3
    answer = answer.replace("M. O.", "M.O.")
    answer = answer.replace("0. 09\u00b0F", "0.09\u00b0F")
    # no changes to no-gating-no-extra-layers experiment 1 and 2.
    # changes to no-gating-no-extra-layers experiment 3.
    answer = answer.replace("Nancy ]. \u2019", "Nancy].\u2019")
    answer = answer.replace("Windows 8. 1", "Windows 8.1")

    #### GQA
    answer = answer.replace("R. J.", "R.J.")
    answer = answer.replace("Amazon. co. uk", "Amazon.co.uk")
    answer = answer.replace("No. 189", "No.189")
    answer = answer.replace("World no. 1 McIlroy", "World no.1 McIlroy")
    answer = answer.replace("J. C. Harvey Jr.", "J.C. Harvey Jr.")
    if idx == 7367:
        answer = answer.replace("E. coli", "E.coli")
    answer = answer.replace("F ( 13. 6\u00b0C", "F (13.6\u00b0C")
    answer = answer.replace("56 .4\u00b0F", "56.4\u00b0F")
    if idx == 5070:
        answer = answer.replace("World No. 2","World No.2")
    #if idx == 2702 or idx == 4347:

    #answer = answer.replace("L. K. Bennett", "L.K.Bennett")


    if answer_old != answer:
        print(f"old_answer: {answer_old}\nanswer_new: {answer}")

    return answer

if __name__ == "__main__":

    iteration = 200000
    exp = 3
    type_ = "no-gating-end"
    file_with_artefacts = "/data/kkno604/NGT_experiments_updated/general_experiments/"+type_+"/exp"+str(exp)+"/Results/get_results_in_domain/ReCoRD/prediction_files/" \
                                                              "iteration"+str(iteration)+".jsonl"
    file_fixed = "/data/kkno604/NGT_experiments_updated/general_experiments/"+type_+"/exp"+str(exp)+"/Results/get_results_in_domain/ReCoRD/prediction_files/" \
                                                              "iteration_fixed"+str(iteration)+".jsonl"

    with open(file_with_artefacts, "r") as f:
        answer_data = [json.loads(line) for line in f]

    for dict_ in answer_data:
        with open(file_fixed, "a") as f:
            answer_ = fix_answer(dict_["label"], dict_["idx"])
            line = {"idx": dict_["idx"], "label": answer_}
            json.dump(line, f)
            f.write('\n')

