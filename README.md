# Fair First-Stage Recommender

This repo contains the code for the empirical evaluation in the paper [Uncertainty Quantification for Fairness in Two-Stage Recommender Systems](https://arxiv.org/abs/2205.15436), 
with an implementation of the union and monotone threshold selection rules. 


### Create Environment
Make sure [conda](https://docs.conda.io/en/latest/) is installed. Run
```angular2html
conda env create -f environment.yml
conda activate fair_first_stage_recommender
```

### Download and Prepare Data

Download [Microsoft Learning to Rank 30k Dataset](https://www.microsoft.com/en-us/research/project/mslr/) (MSLR-WEB30K). 

Create a folder "./data/" and move train.txt, vali.txt, and test.txt in Fold1 to the data folder. 

Run
```angular2html
python ./scripts/prepare_data.py
```

### Run Experiments

On a cluster with [Slurm](https://slurm.schedmd.com/documentation.html) workload manager (you might want to 
change the partitions you would like to use in ./scrpts/exp_utils.py), run
```angular2html
python ./scripts/run_exp_cal_size.py
python ./scripts/run_exp_noise_ratio.py
python ./scripts/run_exp_t_max.py
python ./scripts/run_exp_W_max.py
```

### Plot Figures
Run
```angular2html
python ./scripts/plot_exp_cal_size.py
python ./scripts/plot_exp_noise_ratio.py
python ./scripts/plot_exp_t_max.py
python ./scripts/plot_exp_W_max.py
```
### Bibtex
```angular2html
@InProceedings{wang/joachims/2023/uncertainty,
  title = {Uncertainty Quantification for Fairness in Two-Stage Recommender Systems},
  author = {Wang, Lequn and Joachims, Thorsten},
  booktitle = {ACM Conference on Web Search and Data Mining (WSDM)},
  year= {2023}
}
```