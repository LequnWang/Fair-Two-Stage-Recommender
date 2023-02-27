import pickle
params = {'legend.fontsize': 28,
          'xtick.labelsize': 28,
          'ytick.labelsize': 28,
          'lines.markersize': 15,
          'errorbar.capsize': 8.0,
          }
line_width = 3.0
transparency = 0.2
font_size = 28
capthick = 3.0

algorithm_labels = {
    "CIPW_LB_mono": "CIPW-LB-mono",
    "CIPW_LB_union": "CIPW-LB-union",
    "uncalibrated_individual": "Uncalibrated Individual",
    "uncalibrated_marginal": "Uncalibrated Marginal",
    "IPW": "IPW",
    "platt_scal_individual": "Platt Individual",
    "platt_scal_marginal": "Platt Marginal",
    "platt_scal_pg_individual": "Platt PG Individual",
    "platt_scal_pg_marginal": "Platt PG Marginal",
}
algorithm_colors = {
    "CIPW_LB_mono": "tab:blue",
    "CIPW_LB_union": "tab:red",
    "uncalibrated_individual": "tab:green",
    "uncalibrated_marginal": "tab:brown",
    "IPW": "tab:purple",
    "platt_scal_individual": "tab:cyan",
    "platt_scal_marginal": "tab:orange",
    "platt_scal_pg_individual": "tab:pink",
    "platt_scal_pg_marginal": "olive",
}
algorithm_markers = {
    "CIPW_LB_mono": "s",
    "CIPW_LB_union": "D",
    "uncalibrated_individual": "1",
    "uncalibrated_marginal": "2",
    "IPW": "o",
    "platt_scal_individual": "v",
    "platt_scal_marginal": "^",
    "platt_scal_pg_individual": "<",
    "platt_scal_pg_marginal": ">",
}
metrics = ["num_selected_clicked", "num_relevant_clicked", "constraint_satisfied_clicked",
           "num_selected_not_clicked", "num_relevant_not_clicked", "constraint_satisfied_not_clicked"]


def collect_results(result_path, exp_parameter, algorithm, results):
    with open(result_path, 'rb') as f:
        result = pickle.load(f)
    results[exp_parameter][algorithm]["num_selected_clicked"]["values"].append(result["num_selected_clicked"])
    results[exp_parameter][algorithm]["num_relevant_clicked"]["values"].append(result["num_relevant_clicked"])
    results[exp_parameter][algorithm]["constraint_satisfied_clicked"]["values"].append(
        result["constraint_satisfied_clicked"])
    results[exp_parameter][algorithm]["num_selected_not_clicked"]["values"].append(result["num_selected_not_clicked"])
    results[exp_parameter][algorithm]["num_relevant_not_clicked"]["values"].append(result["num_relevant_not_clicked"])
    results[exp_parameter][algorithm]["constraint_satisfied_not_clicked"]["values"].append(
        result["constraint_satisfied_not_clicked"])
