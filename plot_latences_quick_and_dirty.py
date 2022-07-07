import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

base_dir = os.path.dirname(__file__)
result_dir = os.path.join(base_dir, "results")
csv_dir = os.path.join(result_dir, "csv")
plot_dir = os.path.join(result_dir, "plots")

# make dirs
if not os.path.exists(csv_dir):
    os.makedirs(csv_dir)
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
# batch size

batch_1 = pd.read_csv(os.path.join(csv_dir, 'batch_1.csv'))[:-1]
batch_2 = pd.read_csv(os.path.join(csv_dir, 'batch_2.csv'))[:-1]
batch_4 = pd.read_csv(os.path.join(csv_dir, 'batch_4.csv'))[:-1]
batch_8 = pd.read_csv(os.path.join(csv_dir,'batch_8.csv'))[:-1]
batch_16 = pd.read_csv(os.path.join(csv_dir,'batch_16.csv'))[:-1]
batch_32 = pd.read_csv(os.path.join(csv_dir,'batch_32.csv'))[:-1]
batch_64 = pd.read_csv(os.path.join(csv_dir,'batch_64.csv'))[:-1]
batch_128 = pd.read_csv(os.path.join(csv_dir,'batch_128.csv'))[:-1]

all_files = [batch_1, batch_2, batch_4, batch_8, batch_16, batch_32, batch_64, batch_128]

feature_extraction_cpu = []
embedding_cpu = []
total_cpu = []
search_memory = []
anomaly_map = []
whole_process = []
total_cuda = []
prep_memory_bank = []
for file in all_files:
    feature_extraction_cpu += [file['#1 feature extraction cpu'].mean()]
    embedding_cpu += [file['#3 embedding of features cpu'].mean()]
    search_memory += [file['#5 search with memory bank cpu'].mean()]
    total_cpu += [file['#9 sum cpu'].mean()]
    anomaly_map += [file['#7 anomaly map cpu'].mean()]
    whole_process += [file['#11 whole process cpu'].mean()]
    prep_memory_bank += [file['#13 preparation memory bank'].mean()]
    total_cuda += [file['#12 whole process gpu'].mean()]

fig = plt.figure(figsize=(26,13))

ax = fig.add_subplot()

# line, = ax1.plot([1, 2, 4, 8, 16, 32, 64, 128], feature_extraction_cpu)
# line, = ax2.plot([1, 2, 4, 8, 16, 32, 64, 128], embedding_cpu)
x_vals = [1, 2, 4, 8, 16, 32, 64, 128]
plt.plot(x_vals,feature_extraction_cpu, label = "feature extraction cpu")
plt.plot(x_vals,total_cuda, label = "whole process cuda")
plt.plot(x_vals,embedding_cpu, label = "embeddings cpu")
plt.plot(x_vals,total_cpu, label = "total cpu")
plt.plot(x_vals,search_memory, label = "search with memory bank")
plt.plot(x_vals,anomaly_map, label = "generate anomaly map")
plt.plot(x_vals,whole_process, label = "whole process (outer measurement)")
plt.plot(x_vals,prep_memory_bank, label = "prep memory bank")

ax.set_xscale('log', base=2)

ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

# fig.set_label = 'Batch Size'
# plt.set_label = 'Batch Size'
plt.title('Batch Size Comparison')
plt.xlabel('batch size')
plt.ylabel('elapsed time for each sample [ms] (mean)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plot_dir, 'latences_batch_size.svg'), bbox_inches = 'tight')


# sampling rate

batch_1 = pd.read_csv(os.path.join(csv_dir,'coreset_sampling_rate_1_percent.csv'))[:-1]
batch_2 = pd.read_csv(os.path.join(csv_dir,'coreset_sampling_rate_5_percent.csv'))[:-1]
batch_4 = pd.read_csv(os.path.join(csv_dir,'coreset_sampling_rate_10_percent.csv'))[:-1]
batch_8 = pd.read_csv(os.path.join(csv_dir,'coreset_sampling_rate_50_percent.csv'))[:-1]
batch_16 = pd.read_csv(os.path.join(csv_dir,'coreset_sampling_rate_100_percent.csv'))[:-1]
# batch_32 = pd.read_csv('batch_32.csv')[:-1]
# batch_64 = pd.read_csv('batch_64.csv')[:-1]
# batch_128 = pd.read_csv('batch_128.csv')[:-1]

all_files = [batch_1, batch_2, batch_4, batch_8, batch_16]#, batch_32, batch_64, batch_128]

feature_extraction_cpu = []
embedding_cpu = []
total_cpu = []
search_memory = []
anomaly_map = []
whole_process = []
total_cuda = []
prep_memory_bank = []
for file in all_files:
    feature_extraction_cpu += [file['#1 feature extraction cpu'].mean()]
    embedding_cpu += [file['#3 embedding of features cpu'].mean()]
    search_memory += [file['#5 search with memory bank cpu'].mean()]
    total_cpu += [file['#9 sum cpu'].mean()]
    anomaly_map += [file['#7 anomaly map cpu'].mean()]
    whole_process += [file['#11 whole process cpu'].mean()]
    prep_memory_bank += [file['#13 preparation memory bank'].mean()]
    total_cuda += [file['#12 whole process gpu'].mean()]

fig = plt.figure(figsize=(26,13))

ax = fig.add_subplot()

# line, = ax1.plot([1, 2, 4, 8, 16, 32, 64, 128], feature_extraction_cpu)
# line, = ax2.plot([1, 2, 4, 8, 16, 32, 64, 128], embedding_cpu)
x_vals = [1, 5, 10, 50, 100]
plt.plot(x_vals,feature_extraction_cpu, label = "feature extraction cpu", marker = "x")
plt.plot(x_vals,total_cuda, label = "whole process cuda", marker = "x")
plt.plot(x_vals,embedding_cpu, label = "embeddings cpu", marker = "x")
plt.plot(x_vals,total_cpu, label = "total cpu", marker = "x")
plt.plot(x_vals,search_memory, label = "search with memory bank", marker = "x")
plt.plot(x_vals,anomaly_map, label = "generate anomaly map", marker = "x")
plt.plot(x_vals,whole_process, label = "whole process (outer measurement)", marker = "x")
plt.plot(x_vals,prep_memory_bank, label = "prep memory bank", marker = "x")

ax.set_xscale('log', base=10)

ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter())

# fig.set_label = 'Batch Size'
# plt.set_label = 'Batch Size'
plt.title('Coreset Subsampling Rate Comparison')
plt.xlabel('Rate [%]')
plt.ylabel('elapsed time for each sample [ms] (mean)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'latences_smapling_rate.svg'), bbox_inches = 'tight')


# k of kNN

batch_1 = pd.read_csv(os.path.join(csv_dir,'knn_1.csv'))[:-1]
batch_2 = pd.read_csv(os.path.join(csv_dir,'knn_2.csv'))[:-1]
batch_4 = pd.read_csv(os.path.join(csv_dir,'knn_3.csv'))[:-1]
batch_8 = pd.read_csv(os.path.join(csv_dir,'knn_4.csv'))[:-1]
batch_16 = pd.read_csv(os.path.join(csv_dir,'knn_5.csv'))[:-1]
batch_32 = pd.read_csv(os.path.join(csv_dir,'knn_6.csv'))[:-1]
batch_64 = pd.read_csv(os.path.join(csv_dir,'knn_7.csv'))[:-1]
batch_128 = pd.read_csv(os.path.join(csv_dir,'knn_8.csv'))[:-1]
batch_128_1 = pd.read_csv(os.path.join(csv_dir,'knn_9.csv'))[:-1]
batch_128_2 = pd.read_csv(os.path.join(csv_dir,'knn_10.csv'))[:-1]
batch_128_3 = pd.read_csv(os.path.join(csv_dir,'knn_11.csv'))[:-1]
batch_128_4 = pd.read_csv(os.path.join(csv_dir,'knn_12.csv'))[:-1]
batch_128_5 = pd.read_csv(os.path.join(csv_dir,'knn_13.csv'))[:-1]
batch_128_6 = pd.read_csv(os.path.join(csv_dir,'knn_14.csv'))[:-1]
batch_128_7 = pd.read_csv(os.path.join(csv_dir,'knn_15.csv'))[:-1]
batch_128_8 = pd.read_csv(os.path.join(csv_dir,'knn_20.csv'))[:-1]
batch_128_9 = pd.read_csv(os.path.join(csv_dir,'knn_30.csv'))[:-1]


all_files = [batch_1, batch_2, batch_4, batch_8, batch_16, batch_32, batch_64, batch_128, batch_128_1, batch_128_2, batch_128_3, batch_128_4, batch_128_5, batch_128_6, batch_128_7, batch_128_8, batch_128_9]

feature_extraction_cpu = []
embedding_cpu = []
total_cpu = []
search_memory = []
anomaly_map = []
whole_process = []
total_cuda = []
prep_memory_bank = []
for file in all_files:
    feature_extraction_cpu += [file['#1 feature extraction cpu'].mean()]
    embedding_cpu += [file['#3 embedding of features cpu'].mean()]
    search_memory += [file['#5 search with memory bank cpu'].mean()]
    total_cpu += [file['#9 sum cpu'].mean()]
    anomaly_map += [file['#7 anomaly map cpu'].mean()]
    whole_process += [file['#11 whole process cpu'].mean()]
    prep_memory_bank += [file['#13 preparation memory bank'].mean()]
    total_cuda += [file['#12 whole process gpu'].mean()]

fig = plt.figure(figsize=(26,13))

ax = fig.add_subplot()

# line, = ax1.plot([1, 2, 4, 8, 16, 32, 64, 128], feature_extraction_cpu)
# line, = ax2.plot([1, 2, 4, 8, 16, 32, 64, 128], embedding_cpu)
x_vals = [1, 2, 3,4,5,6,7,8,9,10,11,12,13,14,15,20,30]
plt.plot(x_vals,feature_extraction_cpu, label = "feature extraction cpu", marker = "x")
plt.plot(x_vals,total_cuda, label = "whole process cuda", marker = "x")
plt.plot(x_vals,embedding_cpu, label = "embeddings cpu", marker = "x")
plt.plot(x_vals,total_cpu, label = "total cpu", marker = "x")
plt.plot(x_vals,search_memory, label = "search with memory bank", marker = "x")
plt.plot(x_vals,anomaly_map, label = "generate anomaly map", marker = "x")
plt.plot(x_vals,whole_process, label = "whole process (outer measurement)", marker = "x")
plt.plot(x_vals,prep_memory_bank, label = "prep memory bank", marker = "x")

# ax.set_xscale('log', base=2)

ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

# fig.set_label = 'Batch Size'
# plt.set_label = 'Batch Size'
plt.title('Number of NNs (k)')
plt.xlabel('k')
plt.ylabel('elapsed time for each sample [ms] (mean)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plot_dir, 'latences_k_of_kNN.svg'), bbox_inches = 'tight')


# Input / Load Size

batch_1 = pd.read_csv(os.path.join(csv_dir,'input_load_8.csv'))[:-1]
batch_2 = pd.read_csv(os.path.join(csv_dir,'input_load_16.csv'))[:-1]
batch_4 = pd.read_csv(os.path.join(csv_dir,'input_load_32.csv'))[:-1]
batch_8 = pd.read_csv(os.path.join(csv_dir,'input_load_64.csv'))[:-1]
batch_16 = pd.read_csv(os.path.join(csv_dir,'input_load_128.csv'))[:-1]
batch_32 = pd.read_csv(os.path.join(csv_dir,'batch_256.csv'))[:-1]
batch_64 = pd.read_csv(os.path.join(csv_dir,'batch_320.csv'))[:-1]
# batch_128 = pd.read_csv('batch_128.csv')[:-1]

all_files = [batch_1, batch_2, batch_4, batch_8, batch_16]#, batch_32, batch_64, batch_128]

feature_extraction_cpu = []
embedding_cpu = []
total_cpu = []
search_memory = []
anomaly_map = []
whole_process = []
total_cuda = []
prep_memory_bank = []
for file in all_files:
    feature_extraction_cpu += [file['#1 feature extraction cpu'].mean()]
    embedding_cpu += [file['#3 embedding of features cpu'].mean()]
    search_memory += [file['#5 search with memory bank cpu'].mean()]
    total_cpu += [file['#9 sum cpu'].mean()]
    anomaly_map += [file['#7 anomaly map cpu'].mean()]
    whole_process += [file['#11 whole process cpu'].mean()]
    prep_memory_bank += [file['#13 preparation memory bank'].mean()]
    total_cuda += [file['#12 whole process gpu'].mean()]

fig = plt.figure(figsize=(26,13))

ax = fig.add_subplot()

# line, = ax1.plot([1, 2, 4, 8, 16, 32, 64, 128], feature_extraction_cpu)
# line, = ax2.plot([1, 2, 4, 8, 16, 32, 64, 128], embedding_cpu)
x_vals = [8, 16, 32, 64, 128, 256]#, 320]
plt.plot(x_vals,feature_extraction_cpu, label = "feature extraction cpu", marker = "x")
plt.plot(x_vals,total_cuda, label = "whole process cuda", marker = "x")
plt.plot(x_vals,embedding_cpu, label = "embeddings cpu", marker = "x")
plt.plot(x_vals,total_cpu, label = "total cpu", marker = "x")
plt.plot(x_vals,search_memory, label = "search with memory bank", marker = "x")
plt.plot(x_vals,anomaly_map, label = "generate anomaly map", marker = "x")
plt.plot(x_vals,whole_process, label = "whole process (outer measurement)", marker = "x")
plt.plot(x_vals,prep_memory_bank, label = "prep memory bank", marker = "x")

ax.set_xscale('log', base=2)

ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

# fig.set_label = 'Batch Size'
# plt.set_label = 'Batch Size'
plt.title('Input- (Load-) Size')
plt.xlabel('Size [px]')
plt.ylabel('elapsed time for each sample [ms] (mean)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plot_dir, 'input_load_size.svg'), bbox_inches = 'tight')

