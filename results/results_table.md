Random split: train 32,950 / test 8,238 (test positive rate 0.113). Temporal split: train 32,950 / test 8,238 (train positive rate 0.064, test positive rate 0.308).

| variant             | model                        |   cv_roc_auc |   cv_pr_auc |   roc_auc |   pr_auc |   brier |   threshold |   precision |   recall |    f1 |   accuracy |
|:--------------------|:-----------------------------|-------------:|------------:|----------:|---------:|--------:|------------:|------------:|---------:|------:|-----------:|
| A (no duration)     | Majority baseline            |        0.500 |       0.113 |     0.500 |    0.113 |   0.113 |       0.500 |       0.000 |    0.000 | 0.000 |      0.887 |
| A (no duration)     | Logistic regression          |        0.790 |       0.444 |     0.801 |    0.460 |   0.162 |       0.589 |       0.410 |    0.625 | 0.496 |      0.857 |
| A (no duration)     | Random forest                |        0.792 |       0.468 |     0.809 |    0.491 |   0.075 |       0.136 |       0.416 |    0.643 | 0.505 |      0.858 |
| A (no duration)     | Gradient boosting            |        0.798 |       0.468 |     0.815 |    0.495 |   0.074 |       0.120 |       0.400 |    0.650 | 0.495 |      0.851 |
| A (no duration)     | Gradient boosting + isotonic |      nan     |     nan     |     0.816 |    0.496 |   0.074 |       0.098 |       0.370 |    0.670 | 0.476 |      0.834 |
| B (duration, leaky) | Majority baseline            |        0.500 |       0.113 |     0.500 |    0.113 |   0.113 |       0.500 |       0.000 |    0.000 | 0.000 |      0.887 |
| B (duration, leaky) | Logistic regression          |        0.936 |       0.581 |     0.944 |    0.623 |   0.099 |       0.436 |       0.423 |    0.933 | 0.582 |      0.849 |
| B (duration, leaky) | Random forest                |        0.943 |       0.650 |     0.952 |    0.699 |   0.055 |       0.161 |       0.449 |    0.925 | 0.604 |      0.864 |
| B (duration, leaky) | Gradient boosting            |        0.947 |       0.658 |     0.955 |    0.708 |   0.052 |       0.153 |       0.479 |    0.923 | 0.630 |      0.878 |
| A, temporal split   | Majority baseline            |        0.500 |       0.064 |     0.500 |    0.308 |   0.308 |       0.500 |       0.000 |    0.000 | 0.000 |      0.692 |
| A, temporal split   | Logistic regression          |        0.673 |       0.201 |     0.699 |    0.502 |   0.294 |       0.492 |       0.356 |    0.864 | 0.504 |      0.475 |
| A, temporal split   | Random forest                |        0.676 |       0.221 |     0.743 |    0.538 |   0.209 |       0.077 |       0.390 |    0.918 | 0.547 |      0.531 |
| A, temporal split   | Gradient boosting            |        0.688 |       0.224 |     0.641 |    0.432 |   0.248 |       0.065 |       0.414 |    0.560 | 0.476 |      0.619 |
