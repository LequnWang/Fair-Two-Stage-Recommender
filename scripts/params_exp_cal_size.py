exp_token = "cz"
exp_dir = "./exp_cal_size"
split_size = 500
submit = True
run_all = True

n_cals = [int(1e4), int(2e4), int(5e4), int(1e5), int(2e5), int(5e5)]
noise_ratio_not_clicked = 0
n_runs = 50
k = 5
k_clicked = k * 6.1566 / (6.1566 + 13.9902)
k_not_clicked = k - k_clicked
W_max = 100
t_max = 50
n_cals_label = ["1e4", "2e4", "5e4", "1e5", "2e5", "5e5"]
