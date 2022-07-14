set-ExecutionPolicy RemoteSigned
cd D:\UNI\IIIT_Muen\adapted_PactchCore\PatchCore_anomaly_detection # adapt!f
py train_latences_dev.py --batch_size 56 --avgpool_kernel 3 --avgpool_stride 1 --avgpool_padding 1 --file_name_latences "deault.csv"
py train_latences_dev.py --batch_size 56 --avgpool_kernel 1 --avgpool_stride 1 --avgpool_padding 0 --file_name_latences "no_filter_kernel_1.csv"
py train_latences_dev.py --batch_size 56 --avgpool_kernel 4 --avgpool_stride 1 --avgpool_padding 1 --file_name_latences "default_kernel_4.csv"
py train_latences_dev.py --batch_size 56 --avgpool_kernel 5 --avgpool_stride 1 --avgpool_padding 1 --file_name_latences "default_kernel_5.csv"
py train_latences_dev.py --batch_size 56 --avgpool_kernel 6 --avgpool_stride 1 --avgpool_padding 1 --file_name_latences "default_kernel_6.csv"
py train_latences_dev.py --batch_size 56 --avgpool_kernel 3 --avgpool_stride 2 --avgpool_padding 1 --file_name_latences "default_stride_2.csv"
py train_latences_dev.py --batch_size 56 --avgpool_kernel 4 --avgpool_stride 2 --avgpool_padding 1 --file_name_latences "default_stride_2_kernel_4.csv"
py train_latences_dev.py --batch_size 56 --avgpool_kernel 3 --avgpool_stride 2 --avgpool_padding 0 --file_name_latences "default_stride_2_padding_0.csv"
py train_latences_dev.py --batch_size 56 --avgpool_kernel 5 --avgpool_stride 2 --avgpool_padding 0 --file_name_latences "kernel_5_stride_2_padding_0.csv"
py plot_latences_quick_and_dirty_2.py
shutdown /s