"""
Run the experiments where we vary the maximum thresholds.
"""
import os
from exp_utils import generate_commands, submit_commands
from params_exp_t_max import *


if __name__ == "__main__":
    if not os.path.isdir(exp_dir):
        os.mkdir(exp_dir)
    commands = generate_commands(exp_dir, run_all, [n_cal], [noise_ratio_not_clicked], n_runs, k_clicked, k_not_clicked,
                                 [W_max], t_maxes)
    print(len(commands))
    if submit:
        submit_commands(exp_token, exp_dir, split_size, commands, submit)
