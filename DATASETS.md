# Datasets

The datasets are not redistributed in this repository, because the real-world datasets keep the licenses of their original providers. Download each dataset from its source, save it with the file name below inside a folder named `53_mix_datasets/`, and keep the label column named `class` as given in `new_53_dataset_schemas.json`. The schema file lists, for every dataset, the numeric, ordinal, and nominal attributes and the category order of each ordinal attribute.

| File | n | Numeric | Ordinal | Nominal | Source (URL, access date) |
|---|---|---|---|---|---|
| new_01_automobile.csv | 201 | 15 | 2 | 8 | to be completed |
| new_02_german_credit.csv | 1000 | 7 | 5 | 8 | to be completed |
| new_03_heart_disease.csv | 920 | 6 | 4 | 3 | to be completed |
| new_04_australian_credit.csv | 690 | 6 | 0 | 9 | to be completed |
| new_05_student_performace_mat.csv | 395 | 4 | 11 | 17 | to be completed |
| new_06_telco_customer_churn.csv | 7043 | 3 | 1 | 15 | to be completed (not processed by Bombing-LMD) |
| new_07_cylinder-band.csv | 540 | 19 | 2 | 22 | to be completed |
| new_08_exasens.csv | 399 | 4 | 0 | 2 | to be completed |
| new_09_abalone.csv | 4177 | 7 | 0 | 1 | to be completed |
| new_10_npha-doctor-visits.csv | 714 | 4 | 3 | 7 | to be completed |
| new_11_cirrhosis.csv | 418 | 11 | 1 | 6 | to be completed |
| new_12_caesar.csv | 80 | 2 | 2 | 1 | to be completed |
| new_13_obesity.csv | 2111 | 7 | 3 | 6 | to be completed |
| new_14_heart_failure.csv | 299 | 6 | 0 | 5 | to be completed |
| new_15_stroke-data.csv | 5110 | 3 | 1 | 6 | to be completed |

The synthetic datasets `data-sintesis-01.csv` to `data-sintesis-40.csv` (separated, overlapping, and interleaved structures) are described in the same schema file (`cluster_condition`, `minority_ratio`). The generator script is to be added to this repository.
