import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
from plot_constants import *
from params_exp_noise_ratio import *
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
    metrics = ["num_selected_clicked", "num_relevant_clicked", "constraint_satisfied_clicked",
               "num_selected_not_clicked", "num_relevant_not_clicked", "constraint_satisfied_not_clicked"]
    results = {}
    for noise_ratio_not_clicked in noise_ratios_not_clicked:
        results[noise_ratio_not_clicked] = {}
        for algorithm in algorithms:
            results[noise_ratio_not_clicked][algorithm] = {}
            for metric in metrics:
                results[noise_ratio_not_clicked][algorithm][metric] = {}
                results[noise_ratio_not_clicked][algorithm][metric]["values"] = []

    for noise_ratio_not_clicked in noise_ratios_not_clicked:
        for run in range(n_runs):
            exp_identity_string = "_".join([str(n_cal), str(noise_ratio_not_clicked), str(W_max), str(t_max), str(run)])
            for algorithm in algorithms:
                result_path = os.path.join(exp_dir, exp_identity_string + "_{}_result.pkl".format(algorithm))
                collect_results(result_path, noise_ratio_not_clicked, algorithm, results)

    for noise_ratio_not_clicked in noise_ratios_not_clicked:
        for algorithm in algorithms:
            for metric in metrics:
                results[noise_ratio_not_clicked][algorithm][metric]["mean"] = np.mean(results[noise_ratio_not_clicked][algorithm][metric]["values"])
                results[noise_ratio_not_clicked][algorithm][metric]["std"] = np.std(results[noise_ratio_not_clicked][algorithm][metric]["values"], ddof=1)
                print(noise_ratio_not_clicked, algorithm, metric, results[noise_ratio_not_clicked][algorithm][metric])

    # plotting whether the constraint is satisfied in the clicked group
    handles = []
    for algorithm in algorithms:
        print(algorithm)
        mean_algorithm = np.array([results[noise_ratio_not_clicked][algorithm]["constraint_satisfied_clicked"]["mean"]
                                   for noise_ratio_not_clicked in noise_ratios_not_clicked])
        std_err_algorithm = np.array([results[noise_ratio_not_clicked][algorithm]["constraint_satisfied_clicked"]["std"] / np.sqrt(n_runs)
                                   for noise_ratio_not_clicked in noise_ratios_not_clicked])
        line = axs[0].plot(noise_ratios_not_clicked_label, mean_algorithm, linewidth=line_width, color=algorithm_colors[algorithm],
                           marker=algorithm_markers[algorithm], label=algorithm_labels[algorithm])
        handles = [line[0]] + handles
        axs[0].errorbar(noise_ratios_not_clicked_label, mean_algorithm, std_err_algorithm, linewidth=line_width,
                        color=algorithm_colors[algorithm], marker=algorithm_markers[algorithm],
                        label=algorithm_labels[algorithm])
    # axs[0].yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    axs[0].set_xlabel(" $\epsilon_{\mathrm{disadv}}$\n(a)", fontsize=font_size)
    axs[0].set_ylabel("$\mathrm{ER}_{\mathrm{adv}}$", fontsize=font_size)

    # plotting whether the constraint is satisfied in the not clicked group
    for algorithm in algorithms:
        mean_algorithm = np.array([results[noise_ratio_not_clicked][algorithm]["constraint_satisfied_not_clicked"]["mean"]
                                   for noise_ratio_not_clicked in noise_ratios_not_clicked])
        std_err_algorithm = np.array([results[noise_ratio_not_clicked][algorithm]["constraint_satisfied_not_clicked"]["std"] / np.sqrt(n_runs)
                                   for noise_ratio_not_clicked in noise_ratios_not_clicked])
        axs[1].errorbar(noise_ratios_not_clicked_label, mean_algorithm, std_err_algorithm, linewidth=line_width,
                        color=algorithm_colors[algorithm], marker=algorithm_markers[algorithm],
                        label=algorithm_labels[algorithm])
    # axs[1].yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    axs[1].set_xlabel(" $\epsilon_{\mathrm{disadv}}$\n(b)", fontsize=font_size)
    axs[1].set_ylabel("$\mathrm{ER}_{\mathrm{disadv}}$", fontsize=font_size)

    # plotting the number of selected items in the clicked group
    for algorithm in algorithms:
        mean_algorithm = np.array([results[noise_ratio_not_clicked][algorithm]["num_selected_clicked"]["mean"] for noise_ratio_not_clicked in noise_ratios_not_clicked])
        std_algorithm = np.array([results[noise_ratio_not_clicked][algorithm]["num_selected_clicked"]["std"] for noise_ratio_not_clicked in noise_ratios_not_clicked])
        axs[2].plot(noise_ratios_not_clicked_label, mean_algorithm, linewidth=line_width, color=algorithm_colors[algorithm],
                    marker=algorithm_markers[algorithm], label=algorithm_labels[algorithm])
        axs[2].fill_between(noise_ratios_not_clicked_label, mean_algorithm - std_algorithm, mean_algorithm + std_algorithm,
                            alpha=transparency, color=algorithm_colors[algorithm])
    # axs[2].yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    axs[2].set_xlabel(" $\epsilon_{\mathrm{disadv}}$\n(c)", fontsize=font_size)
    axs[2].set_ylabel("$\mathrm{CSS}_{\mathrm{adv}}$", fontsize=font_size)

    # plotting the number of selected items in the not clicked group
    for algorithm in algorithms:
        mean_algorithm = np.array([results[noise_ratio_not_clicked][algorithm]["num_selected_not_clicked"]["mean"]
                                   for noise_ratio_not_clicked in noise_ratios_not_clicked])
        std_algorithm = np.array([results[noise_ratio_not_clicked][algorithm]["num_selected_not_clicked"]["std"]
                                   for noise_ratio_not_clicked in noise_ratios_not_clicked])
        axs[3].plot(noise_ratios_not_clicked_label, mean_algorithm, linewidth=line_width, color=algorithm_colors[algorithm],
                    marker=algorithm_markers[algorithm], label=algorithm_labels[algorithm])
        axs[3].fill_between(noise_ratios_not_clicked_label, mean_algorithm - std_algorithm, mean_algorithm + std_algorithm,
                            alpha=transparency, color=algorithm_colors[algorithm])
    # axs[3].yaxis.set_major_locator(ticker.MultipleLocator(2))
    axs[3].set_xlabel(" $\epsilon_{\mathrm{disadv}}$ \n(d)", fontsize=font_size)
    axs[3].set_ylabel("$\mathrm{CSS}_{\mathrm{disadv}}$", fontsize=font_size)

    fig.legend(handles=handles, bbox_to_anchor=(0.5, 1.02), loc="upper center", ncol=5)
    plt.tight_layout(rect=[0, 0, 1, 0.81])
    fig.savefig("./plots/exp_noise_ratio.pdf", format="pdf")
