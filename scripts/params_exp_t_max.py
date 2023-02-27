exp_token = "tm"
exp_dir = "./exp_t_max"
split_size = 500
submit = True
run_all = False

n_cal = int(1e5)
noise_ratio_not_clicked = 0
n_runs = 50
k = 5
k_clicked = k * 6.1566 / (6.1566 + 13.9902)
k_not_clicked = k - k_clicked
W_max = 100
t_maxes = [30, 50, 100, 150, 200, 250, 300]
t_maxes_label = ["30", "50", "100", "150", "200", "250", "300"]
