from sklearn.random_projection import SparseRandomProjection
from sampling_methods.kcenter_greedy import kCenterGreedy
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F
from torchvision import transforms
import pytorch_lightning as pl
from PIL import Image
import numpy as np
import argparse
import shutil
import faiss
import torch
import glob
import cv2
import os
import math

from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import nn
import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix
import pickle
from sampling_methods.kcenter_greedy import kCenterGreedy
from sampling_methods.kcenter_greedy_iden import kCenterGreedyIden
from sklearn.random_projection import SparseRandomProjection
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import gaussian_filter

from numpy import genfromtxt


def distance_matrix(x, y=None, p=2):  # pairwise distance of vectors

    y = x if type(y) == type(None) else y

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    dist = torch.pow(x - y, p).sum(2)

    return dist


class NN():  # defining Nearest Neighbour Search

    def __init__(self, X=None, Y=None, p=2):
        self.p = p
        self.train(X, Y)

    def train(self, X, Y):
        self.train_pts = X
        self.train_label = Y

    def __call__(self, x):
        return self.predict(x)

    def predict(self, x):
        if type(self.train_pts) == type(None) or type(self.train_label) == type(None):
            name = self.__class__.__name__
            raise RuntimeError(f"{name} wasn't trained. Need to execute {name}.train() first")

        dist = distance_matrix(x, self.train_pts, self.p) ** (1 / self.p) # Euclidean Distance for p=2
        labels = torch.argmin(dist, dim=1)
        return self.train_label[labels]

class KNN(NN): # K-Nearest-Neighbors, in practical not used ()

    def __init__(self, X=None, Y=None, k=3, p=2):
        self.k = k
        super().__init__(X, Y, p)

    def train(self, X, Y):
        super().train(X, Y)
        if type(Y) != type(None):
            self.unique_labels = self.train_label.unique()

    def predict(self, x):


        # dist = distance_matrix(x, self.train_pts, self.p) ** (1 / self.p)
        dist = torch.cdist(x, self.train_pts, self.p)

        knn = dist.topk(self.k, largest=False)


        return knn

def copy_files(src, dst, ignores=[]):
    src_files = os.listdir(src)
    for file_name in src_files:
        ignore_check = [True for i in ignores if i in file_name]
        if ignore_check:
            continue
        full_file_name = os.path.join(src, file_name)
        if '.py' in full_file_name:
            shutil.copy(full_file_name, os.path.join(dst,file_name))
        # if os.path.isdir(full_file_name):
        #     os.makedirs(os.path.join(dst, file_name), exist_ok=True)
        #     copy_files(full_file_name, os.path.join(dst, file_name), ignores)

def prep_dirs(root):
    # make embeddings dir
    # embeddings_path = os.path.join(root, 'embeddings')
    embeddings_path = os.path.join(root, 'embeddings', args.category)
    os.makedirs(embeddings_path, exist_ok=True)
    # make sample dir
    sample_path = os.path.join(root, 'sample')
    os.makedirs(sample_path, exist_ok=True)
    # make source code record dir & copy
    source_code_save_path = os.path.join(root, 'src')
    os.makedirs(source_code_save_path, exist_ok=True)
    copy_files('./', source_code_save_path, ['.git','.vscode','__pycache__','logs','README','samples','LICENSE']) # copy source code
    return embeddings_path, sample_path, source_code_save_path

def embedding_concat(x, y): # concatenating features from different feature map scales
    # from https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s) # using unfold() to match the size
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z

def reshape_embedding(embedding): # reshape the features as 1-D vectors
    embedding_list = []
    for k in range(embedding.shape[0]):
        for i in range(embedding.shape[2]):
            for j in range(embedding.shape[3]):
                embedding_list.append(embedding[k, :, i, j])
    return embedding_list

#imagenet
mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]

class MVTecDataset(Dataset):
    def __init__(self, root, transform, gt_transform, phase, prop):
        if phase=='train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset(prop) # self.labels => good : 0, anomaly : 1

    def load_dataset(self,prop):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)
        
        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                img_tot_paths=img_tot_paths[:prop]
                gt_tot_paths.extend([0]*len(img_paths))
                gt_tot_paths=gt_tot_paths[:prop]
                tot_labels.extend([0]*len(img_paths))
                tot_labels=tot_labels[:prop]
                tot_types.extend(['good']*len(img_paths))
                tot_types=tot_types[:prop]
                print('#Training Examples:',len(img_tot_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1]*len(img_paths))
                tot_types.extend([defect_type]*len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"
        
        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img) 
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)
        
        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, os.path.basename(img_path[:-4]), img_type

# for the pixel-wise prediction
def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

def heatmap_on_image(heatmap, image):
    if heatmap.shape != image.shape:
        heatmap = cv2.resize(heatmap, (image.shape[0], image.shape[1]))
    out = np.float32(heatmap)/255 + np.float32(image)/255
    out = out / np.max(out)
    return np.uint8(255 * out)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)    


def cal_confusion_matrix(y_true, y_pred_no_thresh, thresh, img_path_list):
    pred_thresh = []
    false_n = []
    false_p = []
    for i in range(len(y_pred_no_thresh)):
        if y_pred_no_thresh[i] > thresh:
            pred_thresh.append(1)
            if y_true[i] == 0:
                false_p.append(img_path_list[i])
        else:
            pred_thresh.append(0)
            if y_true[i] == 1:
                false_n.append(img_path_list[i])

    cm = confusion_matrix(y_true, pred_thresh)
    print(cm)
    print('false positive')
    print(false_p)
    print('false negative')
    print(false_n)
    

class STPM(pl.LightningModule):
    def __init__(self, hparams):
        super(STPM, self).__init__()

        self.save_hyperparameters(hparams)

        self.init_features()
        def hook_t(module, input, output): # hook func for extracting feature maps
            self.features.append(output)

        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True) # load pretrained weight

        if args.pre_weight=='imagenet': # just use the imagenet weight
          print('Using Imagenet weight')
        elif args.pre_weight=='nanodet':
          print('Using Detection weight')
          s_d=torch.load('/content/drive/MyDrive/workspace/synthdata/ELG4960_Out/2/nanodet-m_resnet18_1/model_best/model_best.ckpt')
        elif args.pre_weight=='elg':
          s_d=torch.load('/content/drive/MyDrive/mmclassification/own_data_new/resnet18/best_accuracy_top-1_epoch_300.pth')
        elif args.pre_weight=='mf':
          s_d=torch.load('/content/drive/MyDrive/mmclassification/mf630/resnet18/best_accuracy_top-1_epoch_60.pth')

        if args.pre_weight in ('nanodet','elg','mf'):

          for t in list(s_d['state_dict'].items()):
            for name in self.model.state_dict():
              if 'backbone' in t[0]:
                if args.pre_weight =='nanodet':
                  if t[0][15:] == name:
                    self.model.state_dict()[name].copy_(t[1])
                else:
                  if t[0][9:] == name:
                    self.model.state_dict()[name].copy_(t[1])
        


        for param in self.model.parameters():
            param.requires_grad = False

        self.model.layer2[-1].register_forward_hook(hook_t) # using feature maps of layer 2&3
        self.model.layer3[-1].register_forward_hook(hook_t)
        # self.model.stage2[-1].register_forward_hook(hook_t)
        # self.model.stage3[-1].register_forward_hook(hook_t)
        
        self.criterion = torch.nn.MSELoss(reduction='sum')

        self.init_results_list()
        a = int(args.load_size)
        
        self.data_transforms = transforms.Compose([
                        transforms.Resize((a, a), Image.ANTIALIAS),
                        transforms.ToTensor(),
                        # transforms.CenterCrop(args.input_size),
                        transforms.Normalize(mean=mean_train,
                                            std=std_train)]) # resize input image size + normalization
        self.gt_transforms = transforms.Compose([
                        transforms.Resize((a, a)),
                        # transforms.CenterCrop(args.input_size),
                        transforms.ToTensor()])

        self.inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255], std=[1/0.229, 1/0.224, 1/0.255])

    def init_results_list(self):
        self.gt_list_px_lvl = []
        self.pred_list_px_lvl = []
        self.gt_list_img_lvl = []
        self.pred_list_img_lvl = []
        self.img_path_list = []        

    def init_features(self):
        self.features = []

    def forward(self, x_t):
        self.init_features()
        _ = self.model(x_t)
        return self.features

    def save_anomaly_map(self, anomaly_map, input_img, gt_img, file_name, x_type): # for pixel-wise prediction, can be ignored
        if anomaly_map.shape != input_img.shape:
            anomaly_map = cv2.resize(anomaly_map, (input_img.shape[0], input_img.shape[1]))
        anomaly_map_norm = min_max_norm(anomaly_map)
        anomaly_map_norm_hm = cvt2heatmap(anomaly_map_norm*255)

        # anomaly map on image
        heatmap = cvt2heatmap(anomaly_map_norm*255)
        hm_on_img = heatmap_on_image(heatmap, input_img)

        # save images
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}.jpg'), input_img)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_amap.jpg'), anomaly_map_norm_hm)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_amap_on_img.jpg'), hm_on_img)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_gt.jpg'), gt_img)

    def train_dataloader(self):
        image_datasets = MVTecDataset(root=os.path.join(args.dataset_path,args.category), transform=self.data_transforms, gt_transform=self.gt_transforms, phase='train', prop=args.propotion)
        train_loader = DataLoader(image_datasets, batch_size=args.batch_size, shuffle=True, num_workers=0) #, pin_memory=True)
        return train_loader

    def test_dataloader(self):
        test_datasets = MVTecDataset(root=os.path.join(args.dataset_path,args.category), transform=self.data_transforms, gt_transform=self.gt_transforms, phase='test', prop=452)
        test_loader = DataLoader(test_datasets, batch_size=1, shuffle=False, num_workers=0) #, pin_memory=True) # only work on batch_size=1, now.
        return test_loader

    def configure_optimizers(self):
        return None

    def on_train_start(self):
        self.model.eval() # to stop running_var move (maybe not critical)
        self.embedding_dir_path, self.sample_path, self.source_code_save_path = prep_dirs(args.project_root_path)
        self.embedding_list = []
    
    def on_test_start(self):
        self.embedding_dir_path, self.sample_path, self.source_code_save_path = prep_dirs(args.project_root_path)
        
        self.index = faiss.read_index(os.path.join(self.embedding_dir_path,'index.faiss')) # load memory bank
        # if torch.cuda.is_available():
        #     res = faiss.StandardGpuResources()
        #     self.index = faiss.index_cpu_to_gpu(res, 0 ,self.index)
        self.init_results_list()

        
    def training_step(self, batch, batch_idx): # save locally aware patch features
        x, _, _, file_name, _ = batch
        features = self(x)
        embeddings = []
        for feature in features:
            m = torch.nn.AvgPool2d(3, 1, 1) # using AvgPool2d to calculate local-aware features
            embeddings.append(m(feature))
        embedding = embedding_concat(embeddings[0], embeddings[1])
        self.embedding_list.extend(reshape_embedding(np.array(embedding)))

    def training_epoch_end(self, outputs): 
        total_embeddings = np.array(self.embedding_list)
        # Random projection
        # self.randomprojector = SparseRandomProjection(n_components='auto', eps=0.9) # 'auto' => Johnson-Lindenstrauss lemma
        self.randomprojector = SparseRandomProjection(n_components=120, eps=0.9) # 'auto' => Johnson-Lindenstrauss lemma
        self.randomprojector.fit(total_embeddings)
        # Coreset Subsampling
        # selector = kCenterGreedyIden(total_embeddings,0,0)
        selector = kCenterGreedy(total_embeddings,0,0)
        
        selected_idx = selector.select_batch(model=self.randomprojector, already_selected=[], N=int(total_embeddings.shape[0]*float(args.coreset_sampling_ratio)))
        self.embedding_coreset = total_embeddings[selected_idx]
        
        print('initial embedding size : ', total_embeddings.shape)
        print('final embedding size : ', self.embedding_coreset.shape)
        #faiss
        self.index = faiss.IndexFlatL2(self.embedding_coreset.shape[1])
        self.index.add(self.embedding_coreset) 
        faiss.write_index(self.index,  os.path.join(self.embedding_dir_path,'index.faiss'))

    def prediction_process_core(self, batch, batch_idx):
        '''
        Extracted core process of prediction for better readability.
        '''
        # input // start
        x, gt, label, file_name, x_type = batch # x: feature/input; gt: mask for pixel wise classification ground truth 
        # extract embedding
        features = self(x)
        embeddings = []
        for feature in features: # features: list of feature maps, size: [1,128,8,8] & [1,256,4,4] (first entry --> batch_size = 1)
            m = torch.nn.AvgPool2d(3, 1, 1) # define avg Pooling filter
            embeddings.append(m(feature)) # add to embeddings pooled feature maps, does not change shape
        embedding_ = embedding_concat(embeddings[0], embeddings[1]) # concat two feature maps using unfold() from torch, leads to torch with shape [1, 384, 8, 8]
        embedding_test = np.array(reshape_embedding(np.array(embedding_))) # reshape the features as 1-D vectors, save them as numpy ndarray, shape: [64,384] for batch_size = 1
        score_patches, _ = self.index.search(embedding_test , k=args.n_neighbors) # brutal force search of k nearest neighbours using faiss.IndexFlatL2.search; shape [64,9], memory bank is utilizied     
        anomaly_map = score_patches[:,0].reshape((int(math.sqrt(len(score_patches[:,0]))),int(math.sqrt(len(score_patches[:,0]))))) # shape [8,8]
        ''' 
        MOVED BECAUSE NOT PART OF PREDICTION PROCESS BUT SCORING (?) THEREFORE NOT PART OF LATENCE 
        N_b = score_patches[np.argmax(score_patches[:,0])] # max of each patch
        w = (1 - (np.max(np.exp(N_b))/np.sum(np.exp(N_b)))) # scaling factor in paper
        if math.isnan(w):  
          w = 1.0
        score = w*max(score_patches[:,0]) # Image-level score
        gt_np = gt.cpu().numpy()[0,0].astype(int) # ground_truth as numpy on cpu
        '''
        a = int(args.load_size) # int, 64 
        anomaly_map_resized = cv2.resize(anomaly_map, (a, a)) # [8,8] --> [64,64]
        anomaly_map_resized_blur = gaussian_filter(anomaly_map_resized, sigma=4) # blurr in resized anomaly map; WHY?
        # output // end
        return score_patches, anomaly_map_resized_blur
    
    def test_step(self, batch, batch_idx): # Nearest Neighbour Search
        
        warm_up = 100 # specify how often file should be processed before actual measurment
        reps = 25 # repititions for more meaningful measurements due to averaging
        run_times = [] # initialize timer
        self.file_name_latences = 'latences.txt'
        # warm_up
        for run in range(warm_up):
            _, _ = self.prediction_process_core(batch, batch_idx)
        # measurement
        for rep in range(reps):
            #start timer
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True) # initialize cuda timers
            starter.record() # start
            score_patches, anomaly_map_resized_blur = self.prediction_process_core(batch, batch_idx) # core process
            ender.record() #end
            torch.cuda.synchronize() # wait for gpu sync
            run_times += [starter.elapsed_time(ender)] # in ms, run time of process
        assert len(run_times) == reps, "Something went wrong!"
        latency = sum(run_times) / len(run_times)
        if not math.isnan(latency):
            if os.path.exists(self.file_name_latences):
                with open(self.file_name_latences, 'a') as f:
                    f.write(str(latency) + '\t')
            else:
                with open(self.file_name_latences, 'w') as f:
                    f.write(str(latency) + '\t')
        else:
            print('Latency could not be calculated.')
        # calculating of scores and saving of results
        x, gt, label, file_name, x_type = batch # unpacking of batch
        N_b = score_patches[np.argmax(score_patches[:,0])] # max of each patch
        w = (1 - (np.max(np.exp(N_b))/np.sum(np.exp(N_b)))) # scaling factor in paper
        if math.isnan(w):  
          w = 1.0
        score = w*max(score_patches[:,0])
        gt_np = gt.cpu().numpy()[0,0].astype(int)
        self.gt_list_px_lvl.extend(gt_np.ravel()) # ravel equivalent reshape(-1); flattening of ground_truth pixel wise
        self.pred_list_px_lvl.extend(anomaly_map_resized_blur.ravel()) # flattening of pred pixel wise
        self.gt_list_img_lvl.append(label.cpu().numpy()[0]) # ground_truth for image wise
        self.pred_list_img_lvl.append(score) # image level score appended
        self.img_path_list.extend(file_name) # same for file_name
        # save images
        x = self.inv_normalize(x) # inverse transformation of img
        input_x = cv2.cvtColor(x.permute(0,2,3,1).cpu().numpy()[0]*255, cv2.COLOR_BGR2RGB) # further transformation
        self.save_anomaly_map(anomaly_map_resized_blur, input_x, gt_np*255, file_name[0], x_type[0]) # save of everything

    def test_epoch_end(self, outputs):
        print("Total pixel-level auc-roc score :")
        print(self.gt_list_img_lvl)
        print(self.pred_list_img_lvl)
        pixel_auc = roc_auc_score(self.gt_list_px_lvl, self.pred_list_px_lvl)
        print(pixel_auc)
        print("Total image-level auc-roc score :")
        img_auc = roc_auc_score(self.gt_list_img_lvl, self.pred_list_img_lvl)
        print(img_auc)
        print('test_epoch_end')
        values = {'pixel_auc': pixel_auc, 'img_auc': img_auc}
        self.log_dict(values)
        anomaly_list = []
        normal_list = []
        for i in range(len(self.gt_list_img_lvl)):
            if self.gt_list_img_lvl[i] == 1:
                anomaly_list.append(self.pred_list_img_lvl[i])
            else:
                normal_list.append(self.pred_list_img_lvl[i])

        # thresholding
        # cal_confusion_matrix(self.gt_list_img_lvl, self.pred_list_img_lvl, img_path_list = self.img_path_list, thresh = 0.00097)
        # print()
        with open(args.project_root_path + r'/results.txt', 'a') as f:
            f.write(args.category + ' : ' + str(values) + '\n')

def get_args():
    parser = argparse.ArgumentParser(description='ANOMALYDETECTION')
    parser.add_argument('--phase', choices=['train','test'], default='train')
    parser.add_argument('--dataset_path', default=r'./mvtec_anomaly_detection')#./MVTec') # 'D:\Dataset\mvtec_anomaly_detection')#
    parser.add_argument('--category', default='own')
    parser.add_argument('--num_epochs', default=1) # 1 iteration is enough
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--load_size', default=64) 
    parser.add_argument('--input_size', default=64) # using same input size and load size for our data
    parser.add_argument('--coreset_sampling_ratio', default=0.01)
    parser.add_argument('--project_root_path', default=r'./test') # location to save result
    parser.add_argument('--save_src_code', default=True) 
    parser.add_argument('--save_anomaly_map', default=True)
    parser.add_argument('--n_neighbors', type=int, default=9)
    parser.add_argument('--propotion', type=int, default=452) # number of training samples used, default 452=all samples 
    parser.add_argument('--pre_weight', default='imagenet')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_args()
    trainer = pl.Trainer.from_argparse_args(args, default_root_dir=os.path.join(args.project_root_path, args.category), max_epochs=args.num_epochs, gpus=1) #, check_val_every_n_epoch=args.val_freq,  num_sanity_val_steps=0) # ,fast_dev_run=True)
    model = STPM(hparams=args)
    if args.phase == 'train':
        trainer.fit(model)
        trainer.test(model)
    elif args.phase == 'test':
        trainer.test(model)
        
    all_latencies = genfromtxt(model.file_name_latences, delimiter='\t')
    print(all_latencies)
