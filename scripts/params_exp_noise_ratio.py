exp_token = "nr"
exp_dir = "./exp_noise_ratio"
split_size = 500
submit = True
run_all = True

n_cal = int(1e5)
noise_ratios_not_clicked = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
n_runs = 50
k = 5
k_clicked = k * 6.1566 / (6.1566 + 13.9902)
k_not_clicked = k - k_clicked
W_max = 100
t_max = 50
noise_ratios_not_clicked_label = [str(noise_ratio) for noise_ratio in noise_ratios_not_clicked]
