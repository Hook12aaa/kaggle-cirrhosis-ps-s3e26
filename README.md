# Kaggle Playground Series S3E26 -- Cirrhosis Patient Survival

This is an autonomous run of my [auto-model-trainer](https://github.com/Hook12aaa/auto-model-trainer) plugin. I handed it the objective file and let it drive. It explored 5 architecture classes -- linear, tree_based, catboost, knn, and svm -- across 16 experiments plus a Caruana ensemble. A domain research agent tracked down the original UCI 418-patient cirrhosis dataset and appended it to the training data, which mattered because the rare CL class is only 3.5% of samples. It engineered 22 features, including a Mayo PBC risk score, a cholestasis composite, and a handful of clinical ratios.

## Competition

Multi-class classification predicting patient survival status (C = censored, CL = censored due to liver transplant, D = death). The metric is log_loss (minimize). Training set is 7905 rows, test set is 5271 rows.

Link: https://www.kaggle.com/competitions/playground-series-s3e26

## Results

- Public LB: 0.41130
- Private LB: 0.41651

The winner was a Caruana blend ensemble. The run took 4 exploration rounds and reached tree depth 8. The final Pareto front held exp_001, exp_003, exp_009, exp_015, and exp_ensemble.

## Project Structure

```
.
├── objective.yaml        # the run objective handed to auto-train
├── features.py           # 22 engineered features (Mayo score, cholestasis, ratios)
├── final-report.md       # full evidence trail from the run
├── submission.csv        # ensemble predictions for the test set
└── .auto-trainer/        # experiment tree, convergence config, round logs, scripts
```

## Usage

This solution was produced by the auto-model-trainer plugin for Claude Code:

```
/auto-train objective.yaml
```

The plugin autonomously handles data validation, feature engineering, baseline creation, experiment tree exploration, two-tier convergence detection, ensembling, and the final report with a Kaggle-ready submission.csv.

Plugin: https://github.com/Hook12aaa/auto-model-trainer
