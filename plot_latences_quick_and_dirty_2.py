import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

base_dir = os.path.dirname(__file__)
# base_dir = 'D:\\UNI\\IIIT_Muen\\adapted_PactchCore\\PatchCore_anomaly_detection'
result_dir = os.path.join(base_dir, "results")
csv_dir = os.path.join(result_dir, "csv")
plot_dir = os.path.join(result_dir, "plots")

# make dirs
if not os.path.exists(csv_dir):
    os.makedirs(csv_dir)
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

batch_1 = pd.read_csv(os.path.join(csv_dir, 'default.csv'))
batch_2 = pd.read_csv(os.path.join(csv_dir, 'default_kernel_4.csv'))
batch_4 = pd.read_csv(os.path.join(csv_dir, 'default_kernel_5.csv'))
batch_8 = pd.read_csv(os.path.join(csv_dir, 'default_kernel_6.csv'))
batch_16 = pd.read_csv(os.path.join(csv_dir,'default_stride_2.csv'))
batch_32 = pd.read_csv(os.path.join(csv_dir,'default_stride_2_padding_0.csv'))
# batch_64 = pd.read_csv(os.path.join(csv_dir,'batch_64.csv'))[:-1]
# batch_128 = pd.read_csv(os.path.join(csv_dir,'batch_128.csv'))[:-1]

all_files = [batch_1, batch_2, batch_4, batch_8, batch_16, batch_32]#, batch_64, batch_128]

feature_extraction_cpu = []
embedding_cpu = []
total_cpu = []
search_memory = []
anomaly_map = []
whole_process = []
total_cuda = []
# prep_memory_bank = []
for file in all_files:
    feature_extraction_cpu += [file['#1 feature extraction cpu'].mean()]
    embedding_cpu += [file['#3 embedding of features cpu'].mean()]
    search_memory += [file['#5 search with memory bank cpu'].mean()]
    total_cpu += [file['#9 sum cpu'].mean()]
    anomaly_map += [file['#7 anomaly map cpu'].mean()]
    whole_process += [file['#11 whole process cpu'].mean()]
    # prep_memory_bank += [file['#13 preparation memory bank'].mean()]
    total_cuda += [file['#12 whole process gpu'].mean()]

fig = plt.figure(figsize=(26,13))

ax = fig.add_subplot()

# line, = ax1.plot([1, 2, 4, 8, 16, 32, 64, 128], feature_extraction_cpu)
# line, = ax2.plot([1, 2, 4, 8, 16, 32, 64, 128], embedding_cpu)
x_vals = [64, 49, 36, 25, 16, 9]#, 4]#, 3]#, 8, 16, 32, 64, 128]
plt.plot(x_vals,feature_extraction_cpu, label = "feature extraction cpu", marker = 'x')
plt.plot(x_vals,total_cuda, label = "whole process cuda", marker = 'x')
plt.plot(x_vals,embedding_cpu, label = "embeddings cpu", marker = 'x')
plt.plot(x_vals,total_cpu, label = "total cpu", marker = 'x')
plt.plot(x_vals,search_memory, label = "search with memory bank", marker = 'x')
plt.plot(x_vals,anomaly_map, label = "generate anomaly map", marker = 'x')
plt.plot(x_vals,whole_process, label = "whole process (outer measurement)", marker = 'x')
# plt.plot(x_vals,prep_memory_bank, label = "prep memory bank")

# ax.set_xscale('log', base=2)

ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

# fig.set_label = 'Batch Size'
# plt.set_label = 'Batch Size'
plt.xticks(x_vals,['64','49','36','25','16','9'])
plt.title('Number of Features resulting AvgPool Filter and Embedding (shape: [{x-val}, 384])')
plt.xlabel('Number of Features')
plt.ylabel('elapsed time for each sample [ms] (mean)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plot_dir, 'latences_features.svg'), bbox_inches = 'tight')