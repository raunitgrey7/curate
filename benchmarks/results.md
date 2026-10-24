# Benchmark results (seed 42, 2026-10-24T21:54:36Z, commit e3a7e7a)

## movielens_100k: per_user_temporal (943 evaluated users)

| model | recall@10 | ndcg@10 | hit_rate@10 | coverage@10 | novelty@10 | fit s |
|---|---|---|---|---|---|---|
| popularity | 0.0666 | 0.1214 | 0.5567 | 0.0428 | 1.395 | 0.0 |
| itemknn | 0.1195 | 0.1932 | 0.6946 | 0.1046 | 1.900 | 0.1 |
| bpr | 0.1168 | 0.1873 | 0.7296 | 0.3686 | 2.452 | 6.9 |
| two_tower | 0.1251 | 0.1998 | 0.7275 | 0.4429 | 2.590 | 6.6 |
| two_stage | 0.1371 | 0.2182 | 0.7784 | 0.3514 | 2.452 | 29.2 |

## movielens_100k: global_temporal (69 evaluated users)

| model | recall@10 | ndcg@10 | hit_rate@10 | coverage@10 | novelty@10 | fit s |
|---|---|---|---|---|---|---|
| popularity | 0.0644 | 0.1202 | 0.4058 | 0.0470 | 1.626 | 0.0 |
| itemknn | 0.0391 | 0.1142 | 0.2464 | 0.0820 | 2.060 | 0.1 |
| bpr | 0.0704 | 0.1432 | 0.3913 | 0.1718 | 2.460 | 5.9 |
| two_tower | 0.0803 | 0.1505 | 0.4493 | 0.2099 | 2.829 | 5.8 |
| two_stage | 0.1013 | 0.1619 | 0.4783 | 0.1457 | 2.357 | 24.7 |

## retail_synthetic: per_user_temporal (2000 evaluated users)

| model | recall@10 | ndcg@10 | hit_rate@10 | coverage@10 | novelty@10 | fit s |
|---|---|---|---|---|---|---|
| popularity | 0.0968 | 0.0929 | 0.4475 | 0.0533 | 2.671 | 0.0 |
| itemknn | 0.3399 | 0.3637 | 0.8015 | 0.6750 | 3.709 | 0.0 |
| bpr | 0.2980 | 0.3190 | 0.7710 | 0.9483 | 3.918 | 4.3 |
| two_tower | 0.3019 | 0.3215 | 0.7630 | 0.9867 | 4.017 | 4.6 |
| two_stage | 0.3570 | 0.3853 | 0.8085 | 0.8583 | 3.749 | 22.8 |

## retail_synthetic: global_temporal (1698 evaluated users)

| model | recall@10 | ndcg@10 | hit_rate@10 | coverage@10 | novelty@10 | fit s |
|---|---|---|---|---|---|---|
| popularity | 0.0953 | 0.0730 | 0.3086 | 0.0617 | 2.500 | 0.0 |
| itemknn | 0.3744 | 0.3267 | 0.7103 | 0.7517 | 3.588 | 0.0 |
| bpr | 0.3357 | 0.2908 | 0.6779 | 0.9600 | 3.774 | 5.6 |
| two_tower | 0.3460 | 0.2964 | 0.6843 | 0.9917 | 3.881 | 3.9 |
| two_stage | 0.3951 | 0.3521 | 0.7291 | 0.8933 | 3.633 | 24.6 |

## Counterfactual evaluation (top-10 target policies, 5 percent exploration)

### movielens_100k (50000 logged rounds; reward: 1 if the user rated the item 4 or more in the test window, else 0)

| policy | true value | IPS | SNIPS | DM | DR | IPS ESS |
|---|---|---|---|---|---|---|
| popularity | 0.0708 | 0.0639 (err 0.0069) | 0.0668 (err 0.0040) | 0.0528 (err 0.0180) | 0.0664 (err 0.0045) | 1446 |
| itemknn | 0.1108 | 0.1031 (err 0.0078) | 0.1124 (err 0.0015) | 0.0442 (err 0.0667) | 0.1076 (err 0.0032) | 1122 |
| two_tower | 0.1128 | 0.1070 (err 0.0058) | 0.1159 (err 0.0031) | 0.0305 (err 0.0824) | 0.1087 (err 0.0042) | 789 |
| two_stage | 0.1166 | 0.1110 (err 0.0056) | 0.1176 (err 0.0010) | 0.0350 (err 0.0816) | 0.1136 (err 0.0029) | 845 |

### retail_synthetic (50000 logged rounds; reward: exact click probability from the generator's latent utility)

| policy | true value | IPS | SNIPS | DM | DR | IPS ESS |
|---|---|---|---|---|---|---|
| popularity | 0.1074 | 0.1081 (err 0.0007) | 0.1086 (err 0.0012) | 0.1073 (err 0.0001) | 0.1085 (err 0.0011) | 5582 |
| itemknn | 0.4465 | 0.4473 (err 0.0008) | 0.4445 (err 0.0020) | 0.0623 (err 0.3842) | 0.4438 (err 0.0027) | 1546 |
| two_tower | 0.4131 | 0.4004 (err 0.0127) | 0.4004 (err 0.0127) | 0.0528 (err 0.3603) | 0.3973 (err 0.0158) | 1078 |
| two_stage | 0.4561 | 0.4583 (err 0.0021) | 0.4601 (err 0.0040) | 0.0593 (err 0.3968) | 0.4551 (err 0.0010) | 1473 |

## A/B simulator (alpha 0.05, 400 replications, up to 20000 arrivals, checks every 100)

| dataset | scenario | control hit rate | treatment hit rate | A/A mSPRT FPR | A/A naive peeking FPR | A/A fixed-horizon FPR | A/B mSPRT power | A/B fixed-horizon power | A/B mean n at decision |
|---|---|---|---|---|---|---|---|---|---|
| movielens_100k | popularity top-10 vs two_stage top-10 | 0.4263 | 0.6257 | 0.0075 | 0.4000 | 0.0325 | 1.0000 | 1.0000 | 768 |
| movielens_100k | two_tower top-10 vs two_stage top-10 | 0.5801 | 0.6257 | 0.0050 | 0.4225 | 0.0400 | 1.0000 | 1.0000 | 4966 |
| retail_synthetic | popularity top-10 vs two_stage top-10 | 0.5999 | 0.8947 | 0.0150 | 0.4300 | 0.0375 | 1.0000 | 1.0000 | 401 |
| retail_synthetic | two_tower top-10 vs two_stage top-10 | 0.8706 | 0.8947 | 0.0250 | 0.4000 | 0.0450 | 0.9925 | 1.0000 | 6778 |

## Serving latency (movielens_100k, 200 sequential requests per model, k=10 with explanations)

| model | p50 ms | p95 ms | p99 ms | mean ms | req/s at concurrency 8 |
|---|---|---|---|---|---|
| popularity | 1.16 | 1.39 | 1.48 | 1.18 | 599 |
| itemknn | 1.21 | 1.45 | 1.59 | 1.24 | 569 |
| bpr | 1.18 | 1.41 | 1.78 | 1.21 | 591 |
| two_tower | 1.17 | 1.39 | 1.50 | 1.22 | 594 |
| two_stage | 3.20 | 3.49 | 3.68 | 3.21 | 248 |

Total benchmark time: 150.6 s.
