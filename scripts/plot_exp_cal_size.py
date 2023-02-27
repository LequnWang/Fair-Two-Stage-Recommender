import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
from plot_constants import *
from params_exp_cal_size import *
plt.rcParams.update(params)
plt.rc('font', family='serif')
plt.rcParams["mathtext.fontset"] = "dejavuserif"


if __name__ == "__main__":
    if not os.path.isdir("plots"):
        os.mkdir("plots")
    fig, axs = plt.subplots(1, 4)
    fig.set_size_inches(28, 6.4)
    algorithms = ["uncalibrated_individual", "uncalibrated_marginal", "IPW", "platt_scal_individual",
                  "platt_scal_marginal", "platt_scal_pg_individual",
                  "platt_scal_pg_marginal", "CIPW_LB_union", "CIPW_LB_mono"]
    results = {}
    for n_cal in n_cals:
        results[n_cal] = {}
        for algorithm in algorithms:
            results[n_cal][algorithm] = {}
            for metric in metrics:
                results[n_cal][algorithm][metric] = {}
                results[n_cal][algorithm][metric]["values"] = []

    for n_cal in n_cals:
        for run in range(n_runs):
            exp_identity_string = "_".join([str(n_cal), str(noise_ratio_not_clicked), str(W_max), str(t_max), str(run)])
            for algorithm in algorithms:
                result_path = os.path.join(exp_dir, exp_identity_string + "_{}_result.pkl".format(algorithm))
                collect_results(result_path, n_cal, algorithm, results)

    for n_cal in n_cals:
        for algorithm in algorithms:
            for metric in metrics:
                results[n_cal][algorithm][metric]["mean"] = np.mean(results[n_cal][algorithm][metric]["values"])
                results[n_cal][algorithm][metric]["std"] = np.std(results[n_cal][algorithm][metric]["values"], ddof=1)
                print(n_cal, algorithm, metric, results[n_cal][algorithm][metric])

    # plotting whether the constraint is satisfied in the clicked group
    handles = []
    for algorithm in algorithms:
        print(algorithm)
        mean_algorithm = np.array([results[n_cal][algorithm]["constraint_satisfied_clicked"]["mean"]
                                   for n_cal in n_cals])
        std_err_algorithm = np.array([results[n_cal][algorithm]["constraint_satisfied_clicked"]["std"] / np.sqrt(n_runs)
                                   for n_cal in n_cals])
        line = axs[0].plot(n_cals_label, mean_algorithm, linewidth=line_width, color=algorithm_colors[algorithm],
                           marker=algorithm_markers[algorithm], label=algorithm_labels[algorithm])
        handles = [line[0]] + handles
        axs[0].errorbar(n_cals_label, mean_algorithm, std_err_algorithm, linewidth=line_width,
                        color=algorithm_colors[algorithm], marker=algorithm_markers[algorithm],
                        label=algorithm_labels[algorithm])
    # axs[0].yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    axs[0].set_xlabel(" $m$ \n(a)", fontsize=font_size)
    axs[0].set_ylabel("$\mathrm{ER}_{\mathrm{adv}}$", fontsize=font_size)

    # plotting whether the constraint is satisfied in the not clicked group
    for algorithm in algorithms:
        mean_algorithm = np.array([results[n_cal][algorithm]["constraint_satisfied_not_clicked"]["mean"]
                                   for n_cal in n_cals])
        std_err_algorithm = np.array([results[n_cal][algorithm]["constraint_satisfied_not_clicked"]["std"] / np.sqrt(n_runs)
                                   for n_cal in n_cals])
        axs[1].errorbar(n_cals_label, mean_algorithm, std_err_algorithm, linewidth=line_width,
                        color=algorithm_colors[algorithm], marker=algorithm_markers[algorithm],
                        label=algorithm_labels[algorithm])
    # axs[1].yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    axs[1].set_xlabel(" $m$ \n(b)", fontsize=font_size)
    axs[1].set_ylabel("$\mathrm{ER}_{\mathrm{disadv}}$", fontsize=font_size)

    # plotting the number of selected items in the clicked group
    for algorithm in algorithms:
        mean_algorithm = np.array([results[n_cal][algorithm]["num_selected_clicked"]["mean"] for n_cal in n_cals])
        std_algorithm = np.array([results[n_cal][algorithm]["num_selected_clicked"]["std"] for n_cal in n_cals])
        axs[2].plot(n_cals_label, mean_algorithm, linewidth=line_width, color=algorithm_colors[algorithm],
                    marker=algorithm_markers[algorithm], label=algorithm_labels[algorithm])
        axs[2].fill_between(n_cals_label, mean_algorithm - std_algorithm, mean_algorithm + std_algorithm,
                            alpha=transparency, color=algorithm_colors[algorithm])
    # axs[2].yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    axs[2].set_xlabel(" $m$ \n(c)", fontsize=font_size)
    axs[2].set_ylabel("$\mathrm{CSS}_{\mathrm{adv}}$", fontsize=font_size)
    axs[2].set_ylim(top=10)

    # plotting the number of selected items in the not clicked group
    for algorithm in algorithms:
        mean_algorithm = np.array([results[n_cal][algorithm]["num_selected_not_clicked"]["mean"]
                                   for n_cal in n_cals])
        std_algorithm = np.array([results[n_cal][algorithm]["num_selected_not_clicked"]["std"]
                                   for n_cal in n_cals])
        axs[3].plot(n_cals_label, mean_algorithm, linewidth=line_width, color=algorithm_colors[algorithm],
                    marker=algorithm_markers[algorithm], label=algorithm_labels[algorithm])
        axs[3].fill_between(n_cals_label, mean_algorithm - std_algorithm, mean_algorithm + std_algorithm,
                            alpha=transparency, color=algorithm_colors[algorithm])
    # axs[3].yaxis.set_major_locator(ticker.MultipleLocator(2))
    axs[3].set_xlabel(" $m$ \n(d)", fontsize=font_size)
    axs[3].set_ylabel("$\mathrm{CSS}_{\mathrm{disadv}}$", fontsize=font_size)
    axs[3].set_ylim(top=25)

    fig.legend(handles=handles, bbox_to_anchor=(0.5, 1.02), loc="upper center", ncol=5)
    plt.tight_layout(rect=[0, 0, 1, 0.81])
    fig.savefig("./plots/exp_cal_size.pdf", format="pdf")
