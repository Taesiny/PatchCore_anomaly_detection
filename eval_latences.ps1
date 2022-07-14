﻿set-ExecutionPolicy RemoteSigned
cd D:\UNI\IIIT_Muen\adapted_PactchCore\PatchCore_anomaly_detection # adapt to path where these py files are located
py train_latences.py --batch_size 1 --file_name_latences "batch_1.csv"
py train_latences.py --batch_size 2 --file_name_latences "batch_2.csv"
py train_latences.py --batch_size 4 --file_name_latences "batch_4.csv"
py train_latences.py --batch_size 8 --file_name_latences "batch_8.csv"
py train_latences.py --batch_size 16 --file_name_latences "batch_16.csv"
py train_latences.py --batch_size 32 --file_name_latences "batch_32.csv"
py train_latences.py --batch_size 64 --file_name_latences "batch_64.csv"
py train_latences.py --batch_size 128 --file_name_latences "batch_128.csv"
py train_latences.py --batch_size 1 --load_size 8 --input_size 8 --file_name_latences "input_load_8.csv"
py train_latences.py --batch_size 1 --load_size 16 --input_size 16 --file_name_latences "input_load_16.csv"
py train_latences.py --batch_size 1 --load_size 32 --input_size 32 --file_name_latences "input_load_32.csv"
py train_latences.py --batch_size 1 --load_size 64 --input_size 64 --file_name_latences "input_load_64.csv"
py train_latences.py --batch_size 1 --load_size 128 --input_size 128 --file_name_latences "input_load_128.csv"
py train_latences.py --batch_size 1 --load_size 256 --input_size 256 --file_name_latences "input_load_256.csv"
#py train_latences.py --batch_size 1 --load_size 320 --input_size 320 --file_name_latences "input_load_320.csv"
py train_latences.py --batch_size 1 --coreset_sampling_ratio 0.01 --file_name_latences "coreset_sampling_rate_1_percent.csv"
py train_latences.py --batch_size 1 --coreset_sampling_ratio 0.05 --file_name_latences "coreset_sampling_rate_5_percent.csv"
py train_latences.py --batch_size 1 --coreset_sampling_ratio 0.1 --file_name_latences "coreset_sampling_rate_10_percent.csv"
py train_latences.py --batch_size 1 --coreset_sampling_ratio 0.5 --file_name_latences "coreset_sampling_rate_50_percent.csv"
py train_latences.py --batch_size 1 --coreset_sampling_ratio 1.0 --file_name_latences "coreset_sampling_rate_100_percent.csv"
py train_latences.py --batch_size 1 --n_neighbors 1 --file_name_latences "knn_1.csv"
py train_latences.py --batch_size 1 --n_neighbors 2 --file_name_latences "knn_2.csv"
py train_latences.py --batch_size 1 --n_neighbors 3 --file_name_latences "knn_3.csv"
py train_latences.py --batch_size 1 --n_neighbors 4 --file_name_latences "knn_4.csv"
py train_latences.py --batch_size 1 --n_neighbors 5 --file_name_latences "knn_5.csv"
py train_latences.py --batch_size 1 --n_neighbors 6 --file_name_latences "knn_6.csv"
py train_latences.py --batch_size 1 --n_neighbors 7 --file_name_latences "knn_7.csv"
py train_latences.py --batch_size 1 --n_neighbors 8 --file_name_latences "knn_8.csv"
py train_latences.py --batch_size 1 --n_neighbors 9 --file_name_latences "knn_9.csv"
py train_latences.py --batch_size 1 --n_neighbors 10 --file_name_latences "knn_10.csv"
py train_latences.py --batch_size 1 --n_neighbors 11 --file_name_latences "knn_11.csv"
py train_latences.py --batch_size 1 --n_neighbors 12 --file_name_latences "knn_12.csv"
py train_latences.py --batch_size 1 --n_neighbors 13 --file_name_latences "knn_13.csv"
py train_latences.py --batch_size 1 --n_neighbors 14 --file_name_latences "knn_14.csv"
py train_latences.py --batch_size 1 --n_neighbors 15 --file_name_latences "knn_15.csv"
py train_latences.py --batch_size 1 --n_neighbors 20 --file_name_latences "knn_20.csv"
py train_latences.py --batch_size 1 --n_neighbors 30 --file_name_latences "knn_30.csv"
py plot_latences_quick_and_dirty.py
shutdown /s