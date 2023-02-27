"""
Utils for running the experiments
"""
import os
import random


def generate_commands(exp_dir, run_all, n_cals, noise_ratios_not_clicked, n_runs, k_clicked, k_not_clicked, W_maxes, t_maxes):
    """
    generate a list of commands from the experiment setup
    """
    commands = []
    for n_cal in n_cals:
        for noise_ratio_not_clicked in noise_ratios_not_clicked:
            for W_max in W_maxes:
                for t_max in t_maxes:
                    for run in range(n_runs):
                        exp_identity_string = "_".join([str(n_cal), str(noise_ratio_not_clicked), str(W_max), str(t_max), str(run)])
                        logging_policy_data_path = os.path.join(exp_dir, exp_identity_string + "_logging_policy_data.pkl")
                        user_feedback_simulation_data_path = os.path.join(exp_dir, exp_identity_string +
                                                                          "_user_feedback_simulation_data.pkl")
                        test_data_path = os.path.join(exp_dir, exp_identity_string + "_test_data.pkl")
                        logging_policy_path = os.path.join(exp_dir, exp_identity_string + "_logging_policy.pkl")
                        user_feedback_path = os.path.join(exp_dir, exp_identity_string + "_user_feedback.pkl")
                        shuffle_data_command = "python ./scripts/shuffle_data.py --data_user_feedback_simulation_path {} " \
                                               "--data_logging_policy_path {} --data_test_path " \
                                               "{} ".format(user_feedback_simulation_data_path, logging_policy_data_path,
                                                            test_data_path)
                        train_logging_policy_command = "python ./src/train_LR.py --train_data_path {} --classifier_path {} " \
                                                       "--noise_ratio_not_clicked {} ".format(logging_policy_data_path,
                                                                                              logging_policy_path,
                                                                                              noise_ratio_not_clicked)
                        simulate_user_feedback_command = "python ./scripts/simulate_user_feedback.py --data_path {} " \
                                                         "--logging_policy_path {} --n_queries {} --simulated_data_path " \
                                                         "{} --t_max {}".format(user_feedback_simulation_data_path,
                                                                                logging_policy_path, n_cal, user_feedback_path,
                                                                                t_max)
                        CIPW_LB_mono_result_path = os.path.join(exp_dir, exp_identity_string + "_CIPW_LB_mono_result.pkl")
                        CIPW_LB_mono_command = "python ./src/CIPW_LB_mono.py --data_test_path {} --user_feedback_path {} " \
                                          "--logging_policy_path {} --result_path {} --k_clicked {} --k_not_clicked " \
                                          "{} --W_max {}".format(test_data_path, user_feedback_path, logging_policy_path,
                                                      CIPW_LB_mono_result_path, k_clicked, k_not_clicked, W_max)
                        CIPW_LB_union_result_path = os.path.join(exp_dir, exp_identity_string + "_CIPW_LB_union_result.pkl")
                        CIPW_LB_union_command = "python ./src/CIPW_LB_union.py --data_test_path {} --user_feedback_path {} " \
                                          "--logging_policy_path {} --result_path {} --k_clicked {} --k_not_clicked " \
                                          "{} --W_max {}".format(test_data_path, user_feedback_path, logging_policy_path,
                                                      CIPW_LB_union_result_path, k_clicked, k_not_clicked, W_max)
                        uncalibrated_individual_result_path = os.path.join(exp_dir, exp_identity_string +
                                                                    "_uncalibrated_individual_result.pkl")
                        uncalibrated_individual_command = "python ./src/uncalibrated_individual.py --data_test_path {} " \
                                                   "--logging_policy_path {} --result_path " \
                                                   "{} --k_clicked {} --k_not_clicked {}".format(test_data_path,
                                                                                                 logging_policy_path,
                                                                                                 uncalibrated_individual_result_path,
                                                                                                 k_clicked, k_not_clicked)
                        uncalibrated_marginal_result_path = os.path.join(exp_dir, exp_identity_string +
                                                                  "_uncalibrated_marginal_result.pkl")
                        uncalibrated_marginal_command = "python ./src/uncalibrated_marginal.py --data_test_path {} --user_feedback_path {} " \
                                                 "--logging_policy_path {} --result_path {} --k_clicked {} " \
                                                 "--k_not_clicked {}".format(test_data_path, user_feedback_path,
                                                                            logging_policy_path, uncalibrated_marginal_result_path,
                                                                            k_clicked, k_not_clicked)
                        IPW_result_path = os.path.join(exp_dir, exp_identity_string +
                                                       "_IPW_result.pkl")
                        IPW_command = "python ./src/IPW.py --data_test_path {} --user_feedback_path {} " \
                                                "--logging_policy_path {} --result_path {} --k_clicked {} " \
                                                "--k_not_clicked {}".format(test_data_path, user_feedback_path,
                                                                            logging_policy_path,
                                                                            IPW_result_path, k_clicked,
                                                                            k_not_clicked)
                        platt_scal_individual_result_path = os.path.join(exp_dir, exp_identity_string +
                                                                         "_platt_scal_individual_result.pkl")
                        platt_scal_individual_command = "python ./src/platt_scal_individual.py --data_test_path {} " \
                                                        "--user_feedback_path {} --logging_policy_path {} --result_path {} " \
                                                        "--k_clicked {} --k_not_clicked " \
                                                        "{}".format(test_data_path, user_feedback_path, logging_policy_path,
                                                                    platt_scal_individual_result_path, k_clicked, k_not_clicked)
                        platt_scal_marginal_result_path = os.path.join(exp_dir, exp_identity_string +
                                                                       "_platt_scal_marginal_result.pkl")
                        platt_scal_marginal_command = "python ./src/platt_scal_marginal.py --data_test_path {} " \
                                                      "--user_feedback_path {} --logging_policy_path {} --result_path {} " \
                                                      "--k_clicked {} --k_not_clicked " \
                                                      "{}".format(test_data_path, user_feedback_path, logging_policy_path,
                                                                  platt_scal_marginal_result_path, k_clicked, k_not_clicked)
                        platt_scal_pg_individual_result_path = os.path.join(exp_dir, exp_identity_string +
                                                                         "_platt_scal_pg_individual_result.pkl")
                        platt_scal_pg_individual_command = "python ./src/platt_scal_pg_individual.py --data_test_path {} " \
                                                        "--user_feedback_path {} --logging_policy_path {} --result_path {} " \
                                                        "--k_clicked {} --k_not_clicked " \
                                                        "{}".format(test_data_path, user_feedback_path, logging_policy_path,
                                                                    platt_scal_pg_individual_result_path, k_clicked, k_not_clicked)
                        platt_scal_pg_marginal_result_path = os.path.join(exp_dir, exp_identity_string +
                                                                       "_platt_scal_pg_marginal_result.pkl")
                        platt_scal_pg_marginal_command = "python ./src/platt_scal_pg_marginal.py --data_test_path {} " \
                                                      "--user_feedback_path {} --logging_policy_path {} --result_path {} " \
                                                      "--k_clicked {} --k_not_clicked " \
                                                      "{}".format(test_data_path, user_feedback_path, logging_policy_path,
                                                                  platt_scal_pg_marginal_result_path, k_clicked, k_not_clicked)
                        remove_data_command = "rm {} {} {} {}".format(logging_policy_data_path,
                                                                      user_feedback_simulation_data_path,
                                                                      test_data_path, user_feedback_path)
                        if run_all:
                            exp_commands = [shuffle_data_command, train_logging_policy_command,
                                            simulate_user_feedback_command, CIPW_LB_mono_command, CIPW_LB_union_command,
                                            uncalibrated_individual_command, uncalibrated_marginal_command, IPW_command,
                                            platt_scal_individual_command, platt_scal_marginal_command,
                                            platt_scal_pg_individual_command, platt_scal_pg_marginal_command,
                                            remove_data_command]
                        else:
                            exp_commands = [shuffle_data_command, train_logging_policy_command,
                                            simulate_user_feedback_command, CIPW_LB_mono_command, CIPW_LB_union_command,
                                            remove_data_command]
                        commands.append(exp_commands)
    return commands


def submit_commands(exp_token, exp_dir, split_size, commands, submit):
    """
    submit commands to server
    """
    perm = list(range(len(commands)))
    random.shuffle(perm)
    commands = [commands[idx] for idx in perm]
    split_len = int((len(commands) - 1) / split_size) + 1
    current_idx = 0
    while True:
        stop = 0
        start = current_idx * split_len
        end = (current_idx + 1) * split_len
        if end >= len(commands):
            stop = 1
            end = len(commands)
        with open(os.path.join(exp_dir, "scripts{}.sh".format(current_idx)), "w") as f:
            for exp_commands in commands[start:end]:
                for exp_command in exp_commands:
                    f.write(exp_command + "\n")
        current_idx += 1
        if stop:
            break

    scripts = [os.path.join(exp_dir, "scripts{}.sh".format(idx)) for idx in range(current_idx)]
    cnt = 0
    for script in scripts:
        submission_command = "sbatch --partition=default_partition --requeue -N1 -n1 -c1 --mem=32G " \
                             "-t 72:00:00 -J %s -o %s.o -e %s.e --wrap=\"sh %s\"" % (exp_token + str(cnt), script,
                                                                                     script, script)
        cnt += 1
        if submit:
            os.system(submission_command)
    return
