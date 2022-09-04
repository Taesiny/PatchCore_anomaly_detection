set-ExecutionPolicy RemoteSigned
cd D:\UNI\IIIT_Muen\adapted_PactchCore\PatchCore_anomaly_detection # adapt!
#py train_latences_dev.py --batch_size 14 --feature_maps_selected 2 3 --umap 1 --shrinking_factor 0.5
#py train_latences_dev.py --batch_size 14 --feature_maps_selected 2 3 --umap 1 --shrinking_factor 0.25
#py train_latences_dev.py --batch_size 14 --feature_maps_selected 2 3 --umap 1 --shrinking_factor 0.125
#py train_latences_main.py --batch_size 56 --feature_maps_selected 2 3 --umap 1 --shrinking_factor 0.5 --file_name_latences "umap_without_NN_5.csv"
#py train_latences_main.py --batch_size 56 --feature_maps_selected 2 3 --umap 1 --shrinking_factor 0.25 --file_name_latences "umap_without_NN_25.csv"
#py train_latences_main.py --batch_size 56 --feature_maps_selected 2 3 --umap 1 --shrinking_factor 0.125 --file_name_latences "umap_without_NN_125.csv"
#py train_latences_main.py --batch_size 56 --feature_maps_selected 2 3 --umap 1 --shrinking_factor 0.0625 --file_name_latences "umap_without_NN_0635.csv"
#py train_latences_main.py --batch_size 56 --feature_maps_selected 2 3 --umap 1 --shrinking_factor 0.0625 --file_name_latences "umap_without_NN_0635.csv"

#py train_latences_dev.py --batch_size 14 --feature_maps_selected 2 3 --umap 1 --shrinking_factor 0.03125
#py train_latences_dev.py --batch_size 14 --feature_maps_selected 2 3 --umap 0
py train_latences_main.py --batch_size 56 --feature_maps_selected 2 3 --umap 1 --shrinking_factor 0.25 --umap_metric 'cosine' --file_name_latences "umap_without_NN_cosine_l2.csv"
py train_latences_main.py --batch_size 56 --feature_maps_selected 2 3 --umap 1 --shrinking_factor 0.25 --umap_metric 'haversine' --file_name_latences "umap_without_NN_haversine_l2.csv"
py train_latences_main.py --batch_size 56 --feature_maps_selected 2 3 --umap 1 --shrinking_factor 0.25 --umap_metric 'correlation' --file_name_latences "umap_without_NN_correlation_l2.csv"
py train_latences_main.py --batch_size 56 --feature_maps_selected 2 3 --umap 1 --shrinking_factor 0.25 --umap_metric 'canberra' --file_name_latences "umap_without_NN_canberra_l2.csv"
#py train_latences_main.py --batch_size 56 --feature_maps_selected 2 3 --umap 1 --shrinking_factor 0.25 --umap_metric 'minkowski' --file_name_latences "umap_without_NN_minkowski_l2.csv"
py train_latences_main.py --batch_size 56 --feature_maps_selected 2 3 --umap 1 --shrinking_factor 0.25 --umap_n_neighbors 2 --file_name_latences "umap_without_NN_nn2_l2.csv"
py train_latences_main.py --batch_size 56 --feature_maps_selected 2 3 --umap 1 --shrinking_factor 0.25 --umap_n_neighbors 5 --file_name_latences "umap_without_NN_nn5_l2.csv"
py train_latences_main.py --batch_size 56 --feature_maps_selected 2 3 --umap 1 --shrinking_factor 0.25 --umap_n_neighbors 10 --file_name_latences "umap_without_NN_nn10_l2.csv"
py train_latences_main.py --batch_size 56 --feature_maps_selected 2 3 --umap 1 --shrinking_factor 0.25 --umap_n_neighbors 20 --file_name_latences "umap_without_NN_nn20_l2.csv"
py train_latences_main.py --batch_size 56 --feature_maps_selected 2 3 --umap 1 --shrinking_factor 0.25 --umap_n_neighbors 50 --file_name_latences "umap_without_NN_nn50_l2.csv"
py train_latences_main.py --batch_size 56 --feature_maps_selected 2 3 --umap 1 --shrinking_factor 0.25 --umap_n_neighbors 100 --file_name_latences "umap_without_NN_nn100_l2.csv"
py train_latences_main.py --batch_size 56 --feature_maps_selected 2 3 --umap 1 --shrinking_factor 0.25 --umap_min_dist 0.0 --file_name_latences "umap_without_NN_dist0_l2.csv"
py train_latences_main.py --batch_size 56 --feature_maps_selected 2 3 --umap 1 --shrinking_factor 0.25 --umap_min_dist 0.25 --file_name_latences "umap_without_NN_dist25_l2.csv"
py train_latences_main.py --batch_size 56 --feature_maps_selected 2 3 --umap 1 --shrinking_factor 0.25 --umap_min_dist 0.5 --file_name_latences "umap_without_NN_dist5_l2.csv"
py train_latences_main.py --batch_size 56 --feature_maps_selected 2 3 --umap 1 --shrinking_factor 0.25 --umap_min_dist 0.8 --file_name_latences "umap_without_NN_dist8_l2.csv"

