set-ExecutionPolicy RemoteSigned
cd D:\UNI\IIIT_Muen\adapted_PactchCore\PatchCore_anomaly_detection # adapt!
#py train_latences_dev.py --batch_size 56 --avgpool_kernel 5 --avgpool_stride 1 --avgpool_padding 1 --file_name_latences "kernel_5_padding_1.csv"
#py train_latences_dev.py --batch_size 56 --avgpool_kernel 3 --avgpool_stride 1 --avgpool_padding 0 --file_name_latences "kernel_3_padding_0.csv"
#py train_latences_dev.py --batch_size 56 --avgpool_kernel 7 --avgpool_stride 1 --avgpool_padding 2 --file_name_latences "kernel_7_padding_2.csv"
py train_latences_dev.py --batch_size 14 --feature_maps_selected 1 2 3 4 --file_name_latences "1234.csv"
py train_latences_dev.py --batch_size 14 --feature_maps_selected 1 2 3 --file_name_latences "123.csv"
py train_latences_dev.py --batch_size 14 --feature_maps_selected 2 3 4 --file_name_latences "234.csv"
py train_latences_dev.py --batch_size 14 --feature_maps_selected 2 3 --file_name_latences "23.csv"
py train_latences_dev.py --batch_size 14 --feature_maps_selected 1 2 --file_name_latences "12.csv"
py train_latences_dev.py --batch_size 14 --feature_maps_selected 1 3 --file_name_latences "13.csv"
py train_latences_dev.py --batch_size 14 --feature_maps_selected 1 4 --file_name_latences "14.csv"
py train_latences_dev.py --batch_size 14 --feature_maps_selected 2 4 --file_name_latences "24.csv"
py train_latences_dev.py --batch_size 14 --feature_maps_selected 3 4 --file_name_latences "34.csv"
py train_latences_dev.py --batch_size 14 --feature_maps_selected 1 --file_name_latences "1.csv"
py train_latences_dev.py --batch_size 14 --feature_maps_selected 2 --file_name_latences "2.csv"
py train_latences_dev.py --batch_size 14 --feature_maps_selected 3 --file_name_latences "3.csv"
py train_latences_dev.py --batch_size 14 --feature_maps_selected 4 --file_name_latences "4.csv"

