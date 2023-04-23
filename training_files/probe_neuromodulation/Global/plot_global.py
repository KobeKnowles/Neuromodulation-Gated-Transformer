import matplotlib.pyplot as plt
import numpy as np

def load_global_nmgating_only(filepath):
    # here only one file at a time.
    # return a dict with values.

    tmp = None
    with open(filepath, "r") as f:
        tmp = f.readlines()

    #interval05, interval10, interval15, interval20, interval25, interval30, interval35, interval40, interval45, \
    #interval50, interval55, interval60, interval65, interval70, interval75, interval80, interval85, \
    #interval90, interval95, interval100 = None, None, None, None, None, None, None, None, None, None, \
     #                                     None, None, None, None, None, None, None, None, None, None

    interval_list = [None, None, None, None, None, None, None, None, None, None,
                     None, None, None, None, None, None, None, None, None, None]

    for i in range(20): # 0 - 19
        interval_list[i] = int(tmp[i+1].split()[-1])
        #print(tmp[i+1].split())

    counter = int(tmp[-1].split()[-1])

    for i in range(len(interval_list)):
        interval_list[i] = interval_list[i] / counter # get a percentage here.

    assert round(float(sum(interval_list)), 12) == 1.0, f"The probabilities should sum to 1, they don't! ({sum(interval_list)})"

    return interval_list


def test_load_global_nmgating_only():
    filepath = "/data/kkno604/NGT_experiments_updated/neuromodulation_probe/" \
               "Global/BoolQ/BoolQ_global_iteration200000exp1.jsonl"

    interval_list = load_global_nmgating_only(filepath=filepath)
    print(interval_list)

def average_3_interval_lists(lst1, lst2, lst3):
    assert len(lst1) == len(lst2) and len(lst2) == len(lst3), f"All three lists should be the same length!"
    avg = lst1
    for i in range(len(lst2)):
        avg[i] += lst2[i]
        avg[i] += lst3[i]
        avg[i] = avg[i]/3
    return avg

def average_each_element_of_list(*args):

    list = [0 for i in range(len(args[0]))]
    for list_ in args: assert len(list_) == len(args[0]) #check to see that each list in args is equal in length.
    for list_ in args:
        for i in range(len(list_)):
            list[i] = list[i] + list_[i]
    for i in range(len(list)):
        list[i] = list[i] / len(args)

    return list


def get_global_figure_points(filepath_stem):

    #filepath_stem = "/data/kkno604/NGT_experiments_updated/neuromodulation_probe/Global/"

    interval_list_BoolQ1 = load_global_nmgating_only(
        filepath=filepath_stem+"BoolQ/"+"BoolQ_global_iteration200000exp1.jsonl")
    interval_list_BoolQ2 = load_global_nmgating_only(
        filepath=filepath_stem+"BoolQ/"+"BoolQ_global_iteration200000exp2.jsonl")
    interval_list_BoolQ3 = load_global_nmgating_only(
        filepath=filepath_stem+"BoolQ/"+"BoolQ_global_iteration200000exp3.jsonl")
    interval_list_BoolQ = average_3_interval_lists(interval_list_BoolQ1, interval_list_BoolQ2, interval_list_BoolQ3)

    interval_list_CB1 = load_global_nmgating_only(
        filepath=filepath_stem + "CB/" + "CB_global_iteration200000exp1.jsonl")
    interval_list_CB2 = load_global_nmgating_only(
        filepath=filepath_stem + "CB/" + "CB_global_iteration200000exp2.jsonl")
    interval_list_CB3 = load_global_nmgating_only(
        filepath=filepath_stem + "CB/" + "CB_global_iteration200000exp3.jsonl")
    interval_list_CB = average_3_interval_lists(interval_list_CB1, interval_list_CB2, interval_list_CB3)

    interval_list_COPA1 = load_global_nmgating_only(
        filepath=filepath_stem + "COPA/" + "COPA_global_iteration200000exp1.jsonl")
    interval_list_COPA2 = load_global_nmgating_only(
        filepath=filepath_stem + "COPA/" + "COPA_global_iteration200000exp2.jsonl")
    interval_list_COPA3 = load_global_nmgating_only(
        filepath=filepath_stem + "COPA/" + "COPA_global_iteration200000exp3.jsonl")
    interval_list_COPA = average_3_interval_lists(interval_list_COPA1, interval_list_COPA2, interval_list_COPA3)

    interval_list_MultiRC1 = load_global_nmgating_only(
        filepath=filepath_stem + "MultiRC/" + "MultiRC_global_iteration200000exp1.jsonl")
    interval_list_MultiRC2 = load_global_nmgating_only(
        filepath=filepath_stem + "MultiRC/" + "MultiRC_global_iteration200000exp2.jsonl")
    interval_list_MultiRC3 = load_global_nmgating_only(
        filepath=filepath_stem + "MultiRC/" + "MultiRC_global_iteration200000exp3.jsonl")
    interval_list_MultiRC = average_3_interval_lists(interval_list_MultiRC1, interval_list_MultiRC2, interval_list_MultiRC3)

    interval_list_ReCoRD1 = load_global_nmgating_only(
        filepath=filepath_stem + "ReCoRD/" + "ReCoRD_global_iteration200000exp1.jsonl")
    interval_list_ReCoRD2 = load_global_nmgating_only(
        filepath=filepath_stem + "ReCoRD/" + "ReCoRD_global_iteration200000exp2.jsonl")
    interval_list_ReCoRD3 = load_global_nmgating_only(
        filepath=filepath_stem + "ReCoRD/" + "ReCoRD_global_iteration200000exp3.jsonl")
    interval_list_ReCoRD = average_3_interval_lists(interval_list_ReCoRD1, interval_list_ReCoRD2, interval_list_ReCoRD3)

    interval_list_RTE1 = load_global_nmgating_only(
        filepath=filepath_stem + "RTE/" + "RTE_global_iteration200000exp1.jsonl")
    interval_list_RTE2 = load_global_nmgating_only(
        filepath=filepath_stem + "RTE/" + "RTE_global_iteration200000exp2.jsonl")
    interval_list_RTE3 = load_global_nmgating_only(
        filepath=filepath_stem + "RTE/" + "RTE_global_iteration200000exp3.jsonl")
    interval_list_RTE = average_3_interval_lists(interval_list_RTE1, interval_list_RTE2, interval_list_RTE3)

    interval_list_WiC1 = load_global_nmgating_only(
        filepath=filepath_stem + "WiC/" + "WiC_global_iteration200000exp1.jsonl")
    interval_list_WiC2 = load_global_nmgating_only(
        filepath=filepath_stem + "WiC/" + "WiC_global_iteration200000exp2.jsonl")
    interval_list_WiC3 = load_global_nmgating_only(
        filepath=filepath_stem + "WiC/" + "WiC_global_iteration200000exp3.jsonl")
    interval_list_WiC = average_3_interval_lists(interval_list_WiC1, interval_list_WiC2, interval_list_WiC3)

    interval_list_WSC1 = load_global_nmgating_only(
        filepath=filepath_stem + "WSC/" + "WSC_global_iteration200000exp1.jsonl")
    interval_list_WSC2 = load_global_nmgating_only(
        filepath=filepath_stem + "WSC/" + "WSC_global_iteration200000exp2.jsonl")
    interval_list_WSC3 = load_global_nmgating_only(
        filepath=filepath_stem + "WSC/" + "WSC_global_iteration200000exp3.jsonl")
    interval_list_WSC = average_3_interval_lists(interval_list_WSC1, interval_list_WSC2, interval_list_WSC3)

    interval_list_CQA1 = load_global_nmgating_only(
        filepath=filepath_stem + "CQA/" + "CQA_global_iteration200000exp1.jsonl")
    interval_list_CQA2 = load_global_nmgating_only(
        filepath=filepath_stem + "CQA/" + "CQA_global_iteration200000exp2.jsonl")
    interval_list_CQA3 = load_global_nmgating_only(
        filepath=filepath_stem + "CQA/" + "CQA_global_iteration200000exp3.jsonl")
    interval_list_CQA = average_3_interval_lists(interval_list_CQA1, interval_list_CQA2, interval_list_CQA3)

    interval_list_RACE1 = load_global_nmgating_only(
        filepath=filepath_stem + "RACE/" + "RACE_global_iteration200000exp1.jsonl")
    interval_list_RACE2 = load_global_nmgating_only(
        filepath=filepath_stem + "RACE/" + "RACE_global_iteration200000exp2.jsonl")
    interval_list_RACE3 = load_global_nmgating_only(
        filepath=filepath_stem + "RACE/" + "RACE_global_iteration200000exp3.jsonl")
    interval_list_RACE = average_3_interval_lists(interval_list_RACE1, interval_list_RACE2, interval_list_RACE3)

    interval_list_SciTail1 = load_global_nmgating_only(
        filepath=filepath_stem + "SciTail/" + "SciTail_global_iteration200000exp1.jsonl")
    interval_list_SciTail2 = load_global_nmgating_only(
        filepath=filepath_stem + "SciTail/" + "SciTail_global_iteration200000exp2.jsonl")
    interval_list_SciTail3 = load_global_nmgating_only(
        filepath=filepath_stem + "SciTail/" + "SciTail_global_iteration200000exp3.jsonl")
    interval_list_SciTail = average_3_interval_lists(interval_list_SciTail1, interval_list_SciTail2, interval_list_SciTail3)

    interval_list_MPE1 = load_global_nmgating_only(
        filepath=filepath_stem + "MPE/" + "MPE_global_iteration200000exp1.jsonl")
    interval_list_MPE2 = load_global_nmgating_only(
        filepath=filepath_stem + "MPE/" + "MPE_global_iteration200000exp2.jsonl")
    interval_list_MPE3 = load_global_nmgating_only(
        filepath=filepath_stem + "MPE/" + "MPE_global_iteration200000exp3.jsonl")
    interval_list_MPE = average_3_interval_lists(interval_list_MPE1, interval_list_MPE2, interval_list_MPE3)

    return interval_list_BoolQ, interval_list_CB, interval_list_COPA, interval_list_MultiRC, interval_list_ReCoRD, \
    interval_list_RTE, interval_list_WiC, interval_list_WSC, interval_list_CQA, interval_list_RACE, interval_list_MPE, \
    interval_list_SciTail

def figure_global():
    interval_list_BoolQ, interval_list_CB, interval_list_COPA, interval_list_MultiRC, interval_list_ReCoRD, \
        interval_list_RTE, interval_list_WiC, interval_list_WSC, interval_list_CQA, interval_list_RACE, interval_list_MPE, \
        interval_list_SciTail = get_global_figure_points(
        filepath_stem="/home/kobster/Documents/neuromodulation_probe/Global/")

    x_axis_categories = ["[0-0.05)", "[0.05-0.1)", "[0.1-0.15)", "[0.15-0.2)",
                         "[0.2-0.25)", "[0.25-0.3)", "[0.3-0.35)", "[0.35-0.4)",
                         "[0.4-0.45)", "[0.45-0.5)", "[0.5-0.55)", "[0.55-0.6)",
                         "[0.6-0.65)", "[0.65-0.7)", "[0.7-0.75)", "[0.75-0.8)",
                         "[0.8-0.85)", "[0.85-0.9)", "[0.9-0.95)", "[0.95-1]"]

    plt.plot(x_axis_categories, interval_list_BoolQ, '-o', color="dimgray", label="BoolQ")
    plt.plot(x_axis_categories, interval_list_CB, '-o', color="maroon", label="CB")
    plt.plot(x_axis_categories, interval_list_COPA, '-o', color="lightsalmon", label="COPA")
    plt.plot(x_axis_categories, interval_list_MultiRC, '-o', color="darkorange", label="MultiRC")
    plt.plot(x_axis_categories, interval_list_ReCoRD, '-o', color="yellow", label="ReCoRD")
    plt.plot(x_axis_categories, interval_list_RTE, '-o', color="lawngreen", label="RTE")
    plt.plot(x_axis_categories, interval_list_WiC, '-o', color="darkgreen", label="WiC")
    plt.plot(x_axis_categories, interval_list_WSC, '-o', color="turquoise", label="WSC")
    plt.plot(x_axis_categories, interval_list_CQA, '-o', color="dodgerblue", label="CQA")
    plt.plot(x_axis_categories, interval_list_RACE, '-o', color="darkviolet", label="RACE")
    plt.plot(x_axis_categories, interval_list_MPE, '-o', color="magenta", label="MPE")
    plt.plot(x_axis_categories, interval_list_SciTail, '-o', color="pink", label="SciTail")
    for i in range(len(x_axis_categories)):
        plt.axvline(x=i, color='lightgray', linestyle='--')
    plt.xlabel("Gating Value Regions", fontsize=18)
    plt.ylabel("Probabilities", fontsize=18)
    plt.title("Gating Probabilities Per Region and Dataset", fontsize=22)
    plt.xticks(fontsize=12)
    plt.xticks(fontsize=12)

    # plt.plot(x_BoolQ, y_BoolQ, '-bo', label='BoolQ')
    plt.legend()
    plt.show()

def figure_global_entailment_only():
    interval_list_BoolQ, interval_list_CB, interval_list_COPA, interval_list_MultiRC, interval_list_ReCoRD, \
        interval_list_RTE, interval_list_WiC, interval_list_WSC, interval_list_CQA, interval_list_RACE, interval_list_MPE, \
        interval_list_SciTail = get_global_figure_points(
        filepath_stem="/home/kobster/Documents/neuromodulation_probe/Global/")

    x_axis_categories = ["[0-0.05)", "[0.05-0.1)", "[0.1-0.15)", "[0.15-0.2)",
                         "[0.2-0.25)", "[0.25-0.3)", "[0.3-0.35)", "[0.35-0.4)",
                         "[0.4-0.45)", "[0.45-0.5)", "[0.5-0.55)", "[0.55-0.6)",
                         "[0.6-0.65)", "[0.65-0.7)", "[0.7-0.75)", "[0.75-0.8)",
                         "[0.8-0.85)", "[0.85-0.9)", "[0.9-0.95)", "[0.95-1]"]

    plt.plot(x_axis_categories, interval_list_CB, '-o', color="maroon", label="CB")
    plt.plot(x_axis_categories, interval_list_RTE, '-o', color="lawngreen", label="RTE")
    plt.plot(x_axis_categories, interval_list_MPE, '-o', color="magenta", label="MPE")
    plt.plot(x_axis_categories, interval_list_SciTail, '-o', color="pink", label="SciTail")
    for i in range(len(x_axis_categories)):
        plt.axvline(x=i, color='lightgray', linestyle='--')
    plt.xlabel("Gating Value Regions", fontsize=18)
    plt.ylabel("Probabilities", fontsize=18)
    plt.title("Gating Probabilities Per Region and Dataset", fontsize=22)
    plt.xticks(fontsize=12)
    plt.xticks(fontsize=12)

    # plt.plot(x_BoolQ, y_BoolQ, '-bo', label='BoolQ')
    plt.legend()
    plt.show()

def figure_global_not_entailment_only():
    interval_list_BoolQ, interval_list_CB, interval_list_COPA, interval_list_MultiRC, interval_list_ReCoRD, \
        interval_list_RTE, interval_list_WiC, interval_list_WSC, interval_list_CQA, interval_list_RACE, interval_list_MPE, \
        interval_list_SciTail = get_global_figure_points(
        filepath_stem="/home/kobster/Documents/neuromodulation_probe/Global/")

    x_axis_categories = ["[0-0.05)", "[0.05-0.1)", "[0.1-0.15)", "[0.15-0.2)",
                         "[0.2-0.25)", "[0.25-0.3)", "[0.3-0.35)", "[0.35-0.4)",
                         "[0.4-0.45)", "[0.45-0.5)", "[0.5-0.55)", "[0.55-0.6)",
                         "[0.6-0.65)", "[0.65-0.7)", "[0.7-0.75)", "[0.75-0.8)",
                         "[0.8-0.85)", "[0.85-0.9)", "[0.9-0.95)", "[0.95-1]"]

    plt.plot(x_axis_categories, interval_list_BoolQ, '-o', color="dimgray", label="BoolQ")
    plt.plot(x_axis_categories, interval_list_COPA, '-o', color="lightsalmon", label="COPA")
    plt.plot(x_axis_categories, interval_list_MultiRC, '-o', color="darkorange", label="MultiRC")
    plt.plot(x_axis_categories, interval_list_ReCoRD, '-o', color="yellow", label="ReCoRD")
    plt.plot(x_axis_categories, interval_list_WiC, '-o', color="darkgreen", label="WiC")
    plt.plot(x_axis_categories, interval_list_WSC, '-o', color="turquoise", label="WSC")
    plt.plot(x_axis_categories, interval_list_CQA, '-o', color="dodgerblue", label="CQA")
    plt.plot(x_axis_categories, interval_list_RACE, '-o', color="darkviolet", label="RACE")
    for i in range(len(x_axis_categories)):
        plt.axvline(x=i, color='lightgray', linestyle='--')
    plt.xlabel("Gating Value Regions", fontsize=18)
    plt.ylabel("Probabilities", fontsize=18)
    plt.title("Gating Probabilities Per Region and Dataset", fontsize=22)
    plt.xticks(fontsize=12)
    plt.xticks(fontsize=12)

    # plt.plot(x_BoolQ, y_BoolQ, '-bo', label='BoolQ')
    plt.legend()
    plt.show()

def figure_global_ent_not_entail():
    interval_list_BoolQ, interval_list_CB, interval_list_COPA, interval_list_MultiRC, interval_list_ReCoRD, \
        interval_list_RTE, interval_list_WiC, interval_list_WSC, interval_list_CQA, interval_list_RACE, interval_list_MPE, \
        interval_list_SciTail = get_global_figure_points(
        filepath_stem="/home/kobster/Documents/neuromodulation_probe/Global/")

    x_axis_categories = ["[0-0.05)", "[0.05-0.1)", "[0.1-0.15)", "[0.15-0.2)",
                         "[0.2-0.25)", "[0.25-0.3)", "[0.3-0.35)", "[0.35-0.4)",
                         "[0.4-0.45)", "[0.45-0.5)", "[0.5-0.55)", "[0.55-0.6)",
                         "[0.6-0.65)", "[0.65-0.7)", "[0.7-0.75)", "[0.75-0.8)",
                         "[0.8-0.85)", "[0.85-0.9)", "[0.9-0.95)", "[0.95-1]"]

    ent_avg_list = average_each_element_of_list(interval_list_CB, interval_list_RTE, interval_list_SciTail,
                                                interval_list_MPE)

    not_ent_avg_list = average_each_element_of_list(interval_list_BoolQ, interval_list_COPA, interval_list_MultiRC,
                                                interval_list_ReCoRD, interval_list_WiC, interval_list_WSC,
                                                interval_list_CQA, interval_list_RACE)

    plt.plot(x_axis_categories, ent_avg_list, '-o', color="red", label="Entailment")
    plt.plot(x_axis_categories, not_ent_avg_list, '-o', color="black", label="Not Entailment")
    for i in range(len(x_axis_categories)):
        plt.axvline(x=i, color='lightgray', linestyle='--')
    plt.xlabel("Gating Value Regions", fontsize=18)
    plt.ylabel("Probabilities", fontsize=18)
    plt.title("Gating Probabilities Per Region and Dataset", fontsize=22)
    plt.xticks(fontsize=12)
    plt.xticks(fontsize=12)

    # plt.plot(x_BoolQ, y_BoolQ, '-bo', label='BoolQ')
    plt.legend()
    plt.show()

def figure_global_gating_correct():
    interval_list_BoolQ, interval_list_CB, interval_list_COPA, interval_list_MultiRC, interval_list_ReCoRD, \
        interval_list_RTE, interval_list_WiC, interval_list_WSC, interval_list_CQA, interval_list_RACE, interval_list_MPE, \
        interval_list_SciTail = get_global_figure_points(
        filepath_stem="/home/kobster/Documents/neuromodulation_probe/Global/")

    x_axis_categories = ["[0-0.05)", "[0.05-0.1)", "[0.1-0.15)", "[0.15-0.2)",
                         "[0.2-0.25)", "[0.25-0.3)", "[0.3-0.35)", "[0.35-0.4)",
                         "[0.4-0.45)", "[0.45-0.5)", "[0.5-0.55)", "[0.55-0.6)",
                         "[0.6-0.65)", "[0.65-0.7)", "[0.7-0.75)", "[0.75-0.8)",
                         "[0.8-0.85)", "[0.85-0.9)", "[0.9-0.95)", "[0.95-1]"]

    gating_correct_avg_list = average_each_element_of_list(interval_list_CB, interval_list_RTE, interval_list_SciTail,
                                                interval_list_MPE, interval_list_CQA)

    gating_incorrect_avg_list = average_each_element_of_list(interval_list_BoolQ, interval_list_COPA, interval_list_MultiRC,
                                                interval_list_ReCoRD, interval_list_WiC, interval_list_WSC,
                                                interval_list_RACE)

    plt.plot(x_axis_categories, gating_correct_avg_list, '-o', color="red", label="Correct")
    plt.plot(x_axis_categories, gating_incorrect_avg_list, '-o', color="black", label="Not Correct")
    for i in range(len(x_axis_categories)):
        plt.axvline(x=i, color='lightgray', linestyle='--')
    plt.xlabel("Gating Value Regions", fontsize=18)
    plt.ylabel("Probabilities", fontsize=18)
    plt.title("Gating Probabilities Per Region and Dataset", fontsize=22)
    plt.xticks(fontsize=12)
    plt.xticks(fontsize=12)

    # plt.plot(x_BoolQ, y_BoolQ, '-bo', label='BoolQ')
    plt.legend()
    plt.show()

def figure_global_gating_correct_vs_incorrect_ent():
    c_interval_list_BoolQ, c_interval_list_CB, c_interval_list_COPA, c_interval_list_MultiRC, c_interval_list_ReCoRD, \
        c_interval_list_RTE, c_interval_list_WiC, c_interval_list_WSC, c_interval_list_CQA, c_interval_list_RACE, c_interval_list_MPE, \
        c_interval_list_SciTail = get_global_figure_points(filepath_stem="/home/kobster/Documents/neuromodulation_probe/global_correct/")
    ic_interval_list_BoolQ, ic_interval_list_CB, ic_interval_list_COPA, ic_interval_list_MultiRC, ic_interval_list_ReCoRD, \
        ic_interval_list_RTE, ic_interval_list_WiC, ic_interval_list_WSC, ic_interval_list_CQA, ic_interval_list_RACE, ic_interval_list_MPE, \
        ic_interval_list_SciTail = get_global_figure_points(filepath_stem="/home/kobster/Documents/neuromodulation_probe/global_incorrect/")

    x_axis_categories = ["[0-0.05)", "[0.05-0.1)", "[0.1-0.15)", "[0.15-0.2)",
                         "[0.2-0.25)", "[0.25-0.3)", "[0.3-0.35)", "[0.35-0.4)",
                         "[0.4-0.45)", "[0.45-0.5)", "[0.5-0.55)", "[0.55-0.6)",
                         "[0.6-0.65)", "[0.65-0.7)", "[0.7-0.75)", "[0.75-0.8)",
                         "[0.8-0.85)", "[0.85-0.9)", "[0.9-0.95)", "[0.95-1]"]

    gating_correct_avg_list_not_ent = average_each_element_of_list(c_interval_list_BoolQ, c_interval_list_COPA, c_interval_list_MultiRC, c_interval_list_ReCoRD,
        c_interval_list_WiC, c_interval_list_WSC, c_interval_list_CQA, c_interval_list_RACE)

    gating_incorrect_avg_list_not_ent = average_each_element_of_list(ic_interval_list_BoolQ, ic_interval_list_COPA, ic_interval_list_MultiRC, ic_interval_list_ReCoRD,
        ic_interval_list_WiC, ic_interval_list_WSC, ic_interval_list_CQA, ic_interval_list_RACE)

    gating_correct_avg_list_ent = average_each_element_of_list(c_interval_list_CB, c_interval_list_RTE,
                                                               c_interval_list_MPE, c_interval_list_SciTail)

    gating_incorrect_avg_list_ent = average_each_element_of_list(ic_interval_list_CB, ic_interval_list_RTE,
                                                                 ic_interval_list_MPE, ic_interval_list_SciTail)

    plt.plot(x_axis_categories, gating_correct_avg_list_not_ent, '-o', color="red", label="Correct Not Entailment")
    plt.plot(x_axis_categories, gating_incorrect_avg_list_not_ent, '-o', color="blue", label="Incorrect Not Entailment")
    plt.plot(x_axis_categories, gating_correct_avg_list_ent, '-o', color="orange", label="Correct Entailment")
    plt.plot(x_axis_categories, gating_incorrect_avg_list_ent, '-o', color="purple", label="Incorrect Entailment")
    for i in range(len(x_axis_categories)):
        plt.axvline(x=i, color='lightgray', linestyle='--')
    plt.xlabel("Gating Value Regions", fontsize=18)
    plt.ylabel("Probabilities", fontsize=18)
    plt.title("Gating Probabilities Per Region and Dataset", fontsize=22)
    plt.xticks(fontsize=12)
    plt.xticks(fontsize=12)

    # plt.plot(x_BoolQ, y_BoolQ, '-bo', label='BoolQ')
    plt.legend()
    plt.show()

def figure_global_gating_correct_vs_incorrect():
    c_interval_list_BoolQ, c_interval_list_CB, c_interval_list_COPA, c_interval_list_MultiRC, c_interval_list_ReCoRD, \
        c_interval_list_RTE, c_interval_list_WiC, c_interval_list_WSC, c_interval_list_CQA, c_interval_list_RACE, c_interval_list_MPE, \
        c_interval_list_SciTail = get_global_figure_points(filepath_stem="/home/kobster/Documents/neuromodulation_probe/global_correct/")
    ic_interval_list_BoolQ, ic_interval_list_CB, ic_interval_list_COPA, ic_interval_list_MultiRC, ic_interval_list_ReCoRD, \
        ic_interval_list_RTE, ic_interval_list_WiC, ic_interval_list_WSC, ic_interval_list_CQA, ic_interval_list_RACE, ic_interval_list_MPE, \
        ic_interval_list_SciTail = get_global_figure_points(filepath_stem="/home/kobster/Documents/neuromodulation_probe/global_incorrect/")

    x_axis_categories = ["[0-0.05)", "[0.05-0.1)", "[0.1-0.15)", "[0.15-0.2)",
                         "[0.2-0.25)", "[0.25-0.3)", "[0.3-0.35)", "[0.35-0.4)",
                         "[0.4-0.45)", "[0.45-0.5)", "[0.5-0.55)", "[0.55-0.6)",
                         "[0.6-0.65)", "[0.65-0.7)", "[0.7-0.75)", "[0.75-0.8)",
                         "[0.8-0.85)", "[0.85-0.9)", "[0.9-0.95)", "[0.95-1]"]

    gating_correct_avg_list = average_each_element_of_list(c_interval_list_BoolQ, c_interval_list_CB, c_interval_list_COPA, c_interval_list_MultiRC, c_interval_list_ReCoRD, \
        c_interval_list_RTE, c_interval_list_WiC, c_interval_list_WSC, c_interval_list_CQA, c_interval_list_RACE, c_interval_list_MPE, \
        c_interval_list_SciTail)

    gating_incorrect_avg_list = average_each_element_of_list(ic_interval_list_BoolQ, ic_interval_list_CB, ic_interval_list_COPA, ic_interval_list_MultiRC, ic_interval_list_ReCoRD, \
        ic_interval_list_RTE, ic_interval_list_WiC, ic_interval_list_WSC, ic_interval_list_CQA, ic_interval_list_RACE, ic_interval_list_MPE, \
        ic_interval_list_SciTail)

    plt.plot(x_axis_categories, gating_correct_avg_list, '-o', color="red", label="Correct")
    plt.plot(x_axis_categories, gating_incorrect_avg_list, '-o', color="black", label="Incorrect")
    for i in range(len(x_axis_categories)):
        plt.axvline(x=i, color='lightgray', linestyle='--')
    plt.xlabel("Gating Value Regions", fontsize=18)
    plt.ylabel("Probabilities", fontsize=18)
    plt.title("Gating Probabilities Per Region and Dataset", fontsize=22)
    plt.xticks(fontsize=12)
    plt.xticks(fontsize=12)

    # plt.plot(x_BoolQ, y_BoolQ, '-bo', label='BoolQ')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    #figure_global() #
    #figure_global_entailment_only()
    #figure_global_not_entailment_only()
    #figure_global_ent_not_entail()
    #figure_global_gating_correct() # what is this even used for?
    #figure_global_gating_correct_vs_incorrect()
    figure_global_gating_correct_vs_incorrect_ent()
