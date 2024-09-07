import pickle
import bz2
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy
import numpy as np

import glob
import re
import os

import pandas as pd
from scipy.ndimage import maximum_filter

import seaborn as sns
sns.set(style="white")
sns.set_context("paper", 2.0, {"lines.linewidth": 4})

# TODO clean up code


experiment_dirs = ["./runs"]

environments = ["Hopper-v2",
                "Walker2d-v2",
                "AntTruncatedObs-v2",
                "HumanoidTruncatedObs-v2"
                ]

methods = ["SAC+ToD"]


#
file_name_map = {"list_flip_td.bz2": "policy evaluation",
                 "list_flip_policy_loss.bz2": "policy improvement",
                 "list_flip_q_bias.bz2":  "bias",
                 "list_flip_reinforce.bz2": "return",
                 }
mapenvironmentsname = {"Hopper-v2": "Hopper-v2",
                       "Walker2d-v2": "Walker2d-v2",
                       "AntTruncatedObs-v2": "Ant-v2",
                       "HumanoidTruncatedObs-v2": "Humanoid-v2"}


def _select_worst_case_score(score, metric, number_of_trials=2):
    """
    TODO fill
    :param score:
    :param metric:
    :param number_of_trials:

    :return:
    """
    q_bias_score_t = np.array(score)  # convert np array
    worst_case_score = np.zeros_like(score)[:number_of_trials, :, :, :]  # all trials -> worst-case trials
    for i in range(q_bias_score_t.shape[1]):  # for each epoch
        baseline_score_at_current_epoch = q_bias_score_t[:, i, 0, :]
        sorted_index = np.argsort(baseline_score_at_current_epoch, axis=0).flatten()
        if metric in ["list_q_bias_cleansing", "list_q_bias_cleansing_valid"]:
            sorted_index = sorted_index[::-1]
        worst_scores = q_bias_score_t[sorted_index, i, :, :]
        worst_case_score[:, i, :, :] = worst_scores[:number_of_trials, :, :]
    score = worst_case_score
    return score

def read_bz_results(experiment_dirs, environments, methods):
    """
    Read flip and non-flip results for each method, and environment:
    Environment name x method name  x result name -> [data for each seed]

    :param experiment_dirs: directory path for experimental results
    :param environments: environment names for experiment
    :param methods: method names for experiment

    :rtype: dict (env x method x file name x [data for each seed])
    """
    results = {}
    for env in environments:
        if not (env in results.keys()):
            results[env] = {}
        for method in methods:
            if not (method in results[env].keys()):
                results[env][method] = {}

            # read bz2
            for exp_dir in experiment_dirs:
                performance_files = [p for p in glob.glob(exp_dir + "/" + method + "/**/**/" + "**")
                                     if re.search(".*" + env + ".*bz2", p)]
                for performance_file in performance_files:
                    file = bz2.BZ2File(performance_file, "rb")
                    dataset = pickle.load(file)
                    dataset = np.array(dataset)
                    if performance_file.split("/")[-1] not in results[env][method].keys():
                        results[env][method][performance_file.split("/")[-1]] = []
                    results[env][method][performance_file.split("/")[-1]].append(dataset)
    return results


results_bz2 = read_bz_results(experiment_dirs, environments, methods)




# 1. plot self-influences (average critic loss influence and policy loss)
# plot each results
def plot_influence_positive_ratio_and_colormesh(result, flip_file_name, non_flip_file_name,
                                                scale_x_axis=1,
                                                baseline=None
                                                ):
    """
    TODO fill
    :param result:
    :param flip_file_name:
    :param non_flip_file_name:
    :param scale_x_axis:
    :param baseline:
    """
    # plot influence positive ratio
    plt.clf()
    plt.ticklabel_format(style="sci", axis="y", scilimits=(-1, 1))
    for env in result.keys():
        for method in result[env].keys():
            assert len(result[env].keys()) == 1, "number of method should be one"
            if ((flip_file_name not in result[env][method].keys())
                    or (non_flip_file_name not in result[env][method].keys())):
                continue
            positive_ratios = []
            for seed in range(len(result[env][method][flip_file_name])):
                flip_score = result[env][method][flip_file_name][seed]
                non_flip_score = result[env][method][non_flip_file_name][seed]

                influence = flip_score - non_flip_score
                positive_ratio = np.where(influence > 0.0, 1, 0).mean(axis=-1)
                positive_ratios.append(positive_ratio)
            #
            mean_positive_ratio = np.mean(positive_ratios, axis=0)
            std_positive_ratio = np.std(positive_ratios, axis=0)

            x = np.arange(mean_positive_ratio.shape[0])
            plt.plot(x, mean_positive_ratio, label=mapenvironmentsname[env], zorder=-10)
            line_color = plt.gca().lines[-1].get_color()
            plt.fill_between(x, mean_positive_ratio - std_positive_ratio,
                             mean_positive_ratio + std_positive_ratio,
                             color=line_color, zorder=-10, alpha=0.2)
        plt.ylim([0.0, 1.1])
        plt.title(file_name_map[flip_file_name])
        plt.legend(bbox_to_anchor=(1, 0), loc="lower right", borderaxespad=0.5)
        plt.xlabel("epochs")
        plt.ylabel("Ratio")
        os.makedirs("./figure", exist_ok=True)
        plt.gca().set_rasterization_zorder(0)
        plt.savefig("./figure/" + "ratio_of_influence_" + file_name_map[flip_file_name].replace(" ", "") + ".pdf",
                    bbox_inches='tight')


    # plot contour contour
    for env in result.keys():
        for method in result[env].keys():
            plt.clf()
            plt.ticklabel_format(style="sci", axis="y", scilimits=(-1, 1))
            if ((flip_file_name not in result[env][method].keys())
                    or (non_flip_file_name not in result[env][method].keys())):
                continue
            influences = []
            for seed in range(len(result[env][method][flip_file_name])):
                flip_score = result[env][method][flip_file_name][seed]
                non_flip_score = result[env][method][non_flip_file_name][seed]

                if baseline is not None:
                    baseline_score = result[env][method][baseline][seed][:, 0, 0].reshape((-1, 1))
                    non_flip_score = np.tile(baseline_score, (1, flip_score.shape[1]))

                influence = flip_score - non_flip_score
                if flip_file_name == "list_flip_policy_loss.bz2":
                    influence = - influence
                influences.append(influence)

            mean_influence = np.mean(influences, axis=0)
            for i in range(mean_influence.shape[0]): 
                mean_influence[i] = maximum_filter(mean_influence[i], size=10)
            len_epoch = mean_influence.shape[0]
            num_samples = mean_influence.shape[1]
            x = np.arange(0, int(len_epoch)) * scale_x_axis
            y = np.arange(0, 1, 1.0 / int(num_samples))
            plt.pcolormesh(x, y, np.transpose(mean_influence), zorder=-10, cmap="inferno")
            cbar = plt.colorbar(label="Influence")

            formatter = matplotlib.ticker.ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((-1, 1))
            cbar.ax.yaxis.set_major_formatter(formatter)


            plt.title(mapenvironmentsname[env])
            plt.xlabel("epochs")
            plt.ylabel("normalized experience index")
            plt.gca().set_rasterization_zorder(0)
            plt.savefig("./figure/" + "influence_" + file_name_map[flip_file_name].replace(" ", "") + "_" + env + ".pdf",
                        bbox_inches='tight')


plot_influence_positive_ratio_and_colormesh(results_bz2, "list_flip_td.bz2", "list_non_flip_td.bz2")
plot_influence_positive_ratio_and_colormesh(results_bz2, "list_flip_policy_loss.bz2", "list_non_flip_policy_loss.bz2")







# estimation time.
# read csv result files -> env x method x [{data columns: data}, ... ]
def read_csv_results(experiment_dirs, environments, methods):
    """
    :param experiment_dirs:  directory path for experimental results
    :param environments: environment names for experiment
    :param methods: method names for experiment

    :rtype: dict : env x method x [{data columns: data}, ... ]
    """
    results = {}
    for env in environments:
        if not (env in results.keys()):
            results[env] = {}
        for method in methods:
            if not (method in results[env].keys()):
                results[env][method] = []

            # read bz2
            for exp_dir in experiment_dirs:
                performance_files = [p for p in glob.glob(exp_dir + "/" + method + "/**/**/" + "progress.txt") if
                                     re.search(".*" + env + ".*", p)]
                for performance_file in performance_files:
                    dataset = pd.read_table(performance_file)
                    results[env][method].append(dataset)
    return results


results_csv = read_csv_results(experiment_dirs, environments, methods)

def plot_computational_time(result, plot_baseline_score=True):
    """
    TODO fill
    :param result:
    :param plot_baseline_score:
    """
    num_policy_iteration_per_epoch = 5000 # TODO specify by arg?
    experience_set_size = 5000

    plt.clf()
    plt.ticklabel_format(style="sci", axis="y", scilimits=(-1, 1))
    for env in result.keys():
        assert len(result[env].keys()) == 1, "number of method should be one"
        for method in result[env].keys():
            total_times = []
            for dataset in result[env][method]:
                total_times.append(dataset["Time"].values.flatten())

            diff_total_times = numpy.diff(total_times, axis=1)
            i = 1
            for _ in range(diff_total_times.shape[1]):
                if i % 10 == 0:
                    for j in range(diff_total_times.shape[0]):
                        diff_total_times[j][i-1] = diff_total_times[j][i] 
                i += 1
            total_times = np.zeros_like(total_times)
            total_times[:, 1:] = diff_total_times[:, :]
            total_times = np.cumsum(total_times, axis=1)

            mean_total_time = np.mean(total_times, axis=0)
            std_total_time = np.std(total_times, axis=0)

            x = (np.arange(mean_total_time.shape[0]))
            plt.plot(x, mean_total_time, label=mapenvironmentsname[env], zorder=-10)
            line_color = plt.gca().lines[-1].get_color()
            plt.fill_between(x, mean_total_time - std_total_time,
                             mean_total_time + std_total_time,
                             color=line_color, alpha=0.1,  zorder=-10)

            # plot estimated time of LoO
            if plot_baseline_score:
                average_time_per_pi = np.mean(numpy.diff(mean_total_time)[1:8]) / float(num_policy_iteration_per_epoch)
                total_number_of_pi = np.power(np.arange(mean_total_time.shape[0]) * float(num_policy_iteration_per_epoch), 2) / float(experience_set_size)
                estimated_total_time_loo = total_number_of_pi * average_time_per_pi
                plt.plot(x, estimated_total_time_loo, #label=env,
                         color=line_color, linestyle="--", zorder=-10)

        plt.legend(bbox_to_anchor=(0,1), loc="upper left", borderaxespad=0.5)
        #plt.legend(bbox_to_anchor=(1,0), loc="lower right", borderaxespad=1)

        #plt.title()
        plt.xlabel("epoch")
        plt.ylabel("time (in seconds)")
        plt.gca().set_rasterization_zorder(0)
        if plot_baseline_score:
            plt.savefig("./figure/training_time.pdf", bbox_inches='tight')
        else:
            plt.savefig("./figure/training_time_wo_loo.pdf", bbox_inches='tight')



plot_computational_time(results_csv)
plot_computational_time(results_csv, plot_baseline_score=False)








# plot cleansing result for qbias (validation) and reinforce.
def plot_cleansing_result(result, additional_baseline=None, plot_worst_case=False):
    """
    TODO fill
    :param result:
    :param additional_baseline:
    :param plot_worst_case:
    """
    # plot influence positive ratio
    for metric in ["list_q_bias_cleansing", "list_q_bias_cleansing_valid", "list_reinforce_cleansing"]:
        plt.clf()
        plt.ticklabel_format(style="sci", axis="y", scilimits=(-1, 1))
        for env in result.keys():
            for method in result[env].keys():
                assert len(result[env].keys()) == 1, "number of method should be one"
                if metric + ".bz2" not in result[env][method].keys():
                    continue
                q_bias_score = []
                q_bias_score_valid = []
                for seed in range(len(result[env][method][metric + ".bz2"])):
                    q_bias_score.append(result[env][method][metric + ".bz2"][seed])
                    q_bias_score_valid.append(result[env][method][metric + ".bz2"][seed])

                # worst case screening
                if plot_worst_case:
                    q_bias_score = _select_worst_case_score(q_bias_score, metric, number_of_trials=2)


                # plot post-amendment scores
                if metric in ["list_q_bias_cleansing", "list_q_bias_cleansing_valid"]:
                    min_max_score = np.min(q_bias_score, axis=2)
                else:
                    min_max_score = np.max(q_bias_score, axis=2)
                mean_score_cle = np.mean(min_max_score, axis=0).reshape(-1)
                std_score_cle = np.std(min_max_score, axis=0).reshape(-1)
                x = np.arange(mean_score_cle.shape[0]) * 10
                plt.plot(x, mean_score_cle, label=mapenvironmentsname[env], zorder=-10)
                line_color = plt.gca().lines[-1].get_color()
                plt.fill_between(x, mean_score_cle - std_score_cle,
                                 mean_score_cle + std_score_cle,
                                 color=line_color, zorder=-10, alpha=0.2)

                # plot baseline score
                if additional_baseline[metric] is not None:
                    mean_score = np.mean(q_bias_score, axis=0).reshape(-1, 2)
                    # std_score = np.std(q_bias_score, axis=0).reshape(-1, 2)
                    plt.plot(x, mean_score[:, 0], color=line_color, zorder=-10, linestyle=":")


        plt.legend(bbox_to_anchor=(0, 1), loc="upper left", borderaxespad=0.5)
        #plt.legend(bbox_to_anchor=(1,0), loc="lower right", borderaxespad=1)
        #plt.title()
        plt.xlabel("epoch")
        if metric == "list_reinforce_cleansing":
            plt.ylabel("return")
        else:
            plt.ylabel("bias")
        plt.gca().set_rasterization_zorder(0)
        if plot_worst_case:
            plt.savefig("./figure/cleansing_" + metric + "_worst.pdf", bbox_inches='tight')
        else:
            plt.savefig("./figure/cleansing_" + metric + ".pdf", bbox_inches='tight')




plot_cleansing_result(results_bz2,
                      additional_baseline={"list_q_bias_cleansing": "list_non_flip_q_bias.bz2",
                                           "list_q_bias_cleansing_valid": None,
                                           "list_reinforce_cleansing": "list_non_flip_reinforce.bz2"})
plot_cleansing_result(results_bz2,
                      additional_baseline={"list_q_bias_cleansing": "list_non_flip_q_bias.bz2",
                                           "list_q_bias_cleansing_valid": None,
                                           "list_reinforce_cleansing": "list_non_flip_reinforce.bz2"},
                      plot_worst_case=True)


plot_influence_positive_ratio_and_colormesh(results_bz2,
                                            "list_flip_q_bias.bz2",
                                            "list_non_flip_q_bias.bz2",
                                            scale_x_axis=10,
                                            baseline = "list_q_bias_cleansing.bz2"
                                            )
plot_influence_positive_ratio_and_colormesh(results_bz2,
                                            "list_flip_reinforce.bz2",
                                            "list_non_flip_reinforce.bz2",
                                            scale_x_axis=10,
                                            baseline = "list_reinforce_cleansing.bz2"
                                            )










def plot_num_additional_rollout_amend(result):
    """
    Plot estimated additional rollout for amendment
    :param result:
    """
    plt.clf()
    plt.ticklabel_format(style="sci", axis="y", scilimits=(-1, 1))
    for env in result.keys():
        assert len(result[env].keys()) == 1, "number of method should be one"
        for method in result[env].keys():
            test_ep_len = []
            for dataset in result[env][method]:
                test_ep_len.append(dataset["TestEpLen"].values.flatten())

            test_ep_len = np.array(test_ep_len)
            test_ep_len = test_ep_len[:, ::10]
            x = (np.arange(test_ep_len.shape[1]))
            test_ep_len = test_ep_len * (1.0 + x * 10) * 10.0
            mean_ep_len = np.mean( test_ep_len, axis=0)
            std_ep_len = np.std( test_ep_len, axis=0)
            x = x * 10


            plt.plot(x, mean_ep_len, label=mapenvironmentsname[env], zorder=-10)
            line_color = plt.gca().lines[-1].get_color()
            plt.fill_between(x, mean_ep_len - std_ep_len,
                             mean_ep_len + std_ep_len,
                             color=line_color, alpha=0.1,  zorder=-10)

        plt.legend(bbox_to_anchor=(0,1), loc="upper left", borderaxespad=0.5)
        #plt.legend(bbox_to_anchor=(1,0), loc="lower right", borderaxespad=1)

        #plt.title()
        plt.xlabel("epoch")
        plt.ylabel("environment interaction")
        plt.gca().set_rasterization_zorder(0)
        plt.savefig("./figure/additional_environment_interaction.pdf", bbox_inches='tight')



plot_num_additional_rollout_amend(results_csv)









