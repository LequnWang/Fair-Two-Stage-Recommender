exp_token = "wm"
exp_dir = "./exp_W_max"
split_size = 500
submit = True
run_all = False

n_cal = int(1e5)
noise_ratio_not_clicked = 0
n_runs = 50
k = 5
k_clicked = k * 6.1566 / (6.1566 + 13.9902)
k_not_clicked = k - k_clicked
W_maxes = [10, 20, 40, 60, 80, 100]
t_max = 50
W_maxes_label = ["10", "20", "40", "60", "80", "100"]
