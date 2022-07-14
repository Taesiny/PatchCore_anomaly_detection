set-ExecutionPolicy RemoteSigned
cd D:\UNI\IIIT_Muen\adapted_PactchCore\PatchCore_anomaly_detection
py train_latences_dev.py --batch_size 1 --avgpool_kernel 3 --avgpool_stride 1 --avgpool_padding 1 --file_name_latences "stride_1.csv"
py train_latences_dev.py --batch_size 1 --avgpool_kernel 3 --avgpool_stride 2 --avgpool_padding 1 --file_name_latences "stride_2.csv"
py train_latences_dev.py --batch_size 1 --avgpool_kernel 3 --avgpool_stride 3 --avgpool_padding 1 --file_name_latences "stride_3.csv"
py train_latences_dev.py --batch_size 1 --avgpool_kernel 2 --avgpool_stride 1 --avgpool_padding 1 --file_name_latences "kernel_2.csv"
py train_latences_dev.py --batch_size 1 --avgpool_kernel 3 --avgpool_stride 1 --avgpool_padding 1 --file_name_latences "kernel_3.csv"
py train_latences_dev.py --batch_size 1 --avgpool_kernel 4 --avgpool_stride 1 --avgpool_padding 1 --file_name_latences "kernel_4.csv"
py train_latences_dev.py --batch_size 1 --avgpool_kernel 5 --avgpool_stride 1 --avgpool_padding 1 --file_name_latences "kernel_5.csv"
