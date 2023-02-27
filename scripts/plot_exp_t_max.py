import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
from plot_constants import *
from params_exp_t_max import *
plt.rcParams.update(params)
plt.rc('font', family='serif')
plt.rcParams["mathtext.fontset"] = "dejavuserif"


if __name__ == "__main__":
    if not os.path.isdir("plots"):
        os.mkdir("plots")
    fig, axs = plt.subplots(1, 4)
    fig.set_size_inches(28, 5.6)
    algorithms = ["CIPW_LB_mono", "CIPW_LB_union"]
    metrics = ["num_selected_clicked", "num_relevant_clicked", "constraint_satisfied_clicked",
               "num_selected_not_clicked", "num_relevant_not_clicked", "constraint_satisfied_not_clicked"]
    results = {}
    for t_max in t_maxes:
        results[t_max] = {}
        for algorithm in algorithms:
            results[t_max][algorithm] = {}
            for metric in metrics:
                results[t_max][algorithm][metric] = {}
                results[t_max][algorithm][metric]["values"] = []

    for t_max in t_maxes:
        for run in range(n_runs):
            exp_identity_string = "_".join([str(n_cal), str(noise_ratio_not_clicked), str(W_max), str(t_max), str(run)])
            for algorithm in algorithms:
                result_path = os.path.join(exp_dir, exp_identity_string + "_{}_result.pkl".format(algorithm))
                collect_results(result_path, t_max, algorithm, results)

    for t_max in t_maxes:
        for algorithm in algorithms:
            for metric in metrics:
                results[t_max][algorithm][metric]["mean"] = np.mean(results[t_max][algorithm][metric]["values"])
                results[t_max][algorithm][metric]["std"] = np.std(results[t_max][algorithm][metric]["values"], ddof=1)
                print(t_max, algorithm, metric, results[t_max][algorithm][metric])

    # plotting whether the constraint is satisfied in the clicked group
    handles = []
    for algorithm in algorithms:
        print(algorithm)
        mean_algorithm = np.array([results[t_max][algorithm]["constraint_satisfied_clicked"]["mean"]
                                   for t_max in t_maxes])
        std_err_algorithm = np.array([results[t_max][algorithm]["constraint_satisfied_clicked"]["std"] / np.sqrt(n_runs)
                                   for t_max in t_maxes])
        line = axs[0].plot(t_maxes_label, mean_algorithm, linewidth=line_width, color=algorithm_colors[algorithm],
                           marker=algorithm_markers[algorithm], label=algorithm_labels[algorithm])
        handles.append(line[0])
        axs[0].errorbar(t_maxes_label, mean_algorithm, std_err_algorithm, linewidth=line_width,
                        color=algorithm_colors[algorithm], marker=algorithm_markers[algorithm],
                        label=algorithm_labels[algorithm])
    # axs[0].yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    axs[0].set_xlabel(" $t^{\mathrm{max}}_{\mathrm{adv}} = t^{\mathrm{max}}_{\mathrm{disadv}}$\n(a)", fontsize=font_size)
    axs[0].set_ylabel("$\mathrm{ER}_{\mathrm{adv}}$", fontsize=font_size)

    # plotting whether the constraint is satisfied in the not clicked group
    for algorithm in algorithms:
        mean_algorithm = np.array([results[t_max][algorithm]["constraint_satisfied_not_clicked"]["mean"]
                                   for t_max in t_maxes])
        std_err_algorithm = np.array([results[t_max][algorithm]["constraint_satisfied_not_clicked"]["std"] / np.sqrt(n_runs)
                                   for t_max in t_maxes])
        axs[1].errorbar(t_maxes_label, mean_algorithm, std_err_algorithm, linewidth=line_width,
                        color=algorithm_colors[algorithm], marker=algorithm_markers[algorithm],
                        label=algorithm_labels[algorithm])
    # axs[1].yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    axs[1].set_xlabel(" $t^{\mathrm{max}}_{\mathrm{adv}} = t^{\mathrm{max}}_{\mathrm{disadv}}$\n(b)", fontsize=font_size)
    axs[1].set_ylabel("$\mathrm{ER}_{\mathrm{disadv}}$", fontsize=font_size)

    # plotting the number of selected items in the clicked group
    for algorithm in algorithms:
        mean_algorithm = np.array([results[t_max][algorithm]["num_selected_clicked"]["mean"] for t_max in t_maxes])
        std_algorithm = np.array([results[t_max][algorithm]["num_selected_clicked"]["std"] for t_max in t_maxes])
        axs[2].plot(t_maxes_label, mean_algorithm, linewidth=line_width, color=algorithm_colors[algorithm],
                    marker=algorithm_markers[algorithm], label=algorithm_labels[algorithm])
        axs[2].fill_between(t_maxes_label, mean_algorithm - std_algorithm, mean_algorithm + std_algorithm,
                            alpha=transparency, color=algorithm_colors[algorithm])
    # axs[2].yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    axs[2].set_xlabel(" $t^{\mathrm{max}}_{\mathrm{adv}} = t^{\mathrm{max}}_{\mathrm{disadv}}$\n(c)", fontsize=font_size)
    axs[2].set_ylabel("$\mathrm{CSS}_{\mathrm{adv}}$", fontsize=font_size)

    # plotting the number of selected items in the not clicked group
    for algorithm in algorithms:
        mean_algorithm = np.array([results[t_max][algorithm]["num_selected_not_clicked"]["mean"]
                                   for t_max in t_maxes])
        std_algorithm = np.array([results[t_max][algorithm]["num_selected_not_clicked"]["std"]
                                   for t_max in t_maxes])
        axs[3].plot(t_maxes_label, mean_algorithm, linewidth=line_width, color=algorithm_colors[algorithm],
                    marker=algorithm_markers[algorithm], label=algorithm_labels[algorithm])
        axs[3].fill_between(t_maxes_label, mean_algorithm - std_algorithm, mean_algorithm + std_algorithm,
                            alpha=transparency, color=algorithm_colors[algorithm])
    # axs[3].yaxis.set_major_locator(ticker.MultipleLocator(2))
    axs[3].set_xlabel(" $t^{\mathrm{max}}_{\mathrm{adv}} = t^{\mathrm{max}}_{\mathrm{disadv}}$\n(d)", fontsize=font_size)
    axs[3].set_ylabel("$\mathrm{CSS}_{\mathrm{disadv}}$", fontsize=font_size)

    fig.legend(handles=handles, bbox_to_anchor=(0.5, 1.02), loc="upper center", ncol=4)
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig("./plots/exp_t_max.pdf", format="pdf")
