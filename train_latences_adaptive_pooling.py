from typing import Tuple
from zmq import EVENT_CLOSE_FAILED
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
import pandas as pd
import argparse
import shutil
import faiss
import torch
import glob
import cv2
import os
import math
import time
from enum import Enum
from torch import nn
from torch.optim import Adam
import pickle
from sampling_methods.kcenter_greedy_iden import kCenterGreedyIden
from sklearn.random_projection import SparseRandomProjection
from sklearn.neighbors import NearestNeighbors
import umap
from numpy import genfromtxt
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils import data

class Acceler(Enum):
    '''
    lil enum for ensuring only valid accelerators are chosen.
    '''
    gpu = 1
    cpu = 2

def distance_matrix(x, y=None, p=2):  # pairwise distance of vectors

    y = x if type(y) == type(None) else y

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    dist = torch.pow(x - y, p).sum(2)

    return dist

################################## not used ############################################
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
################################## not used ############################################

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


################################### UMAP MAPPER ################################################ 
# Feed-Forward-NN that learns mapping found by UMAP for fast dim reduction while inference 
################################### UMAP MAPPER ################################################
class Umuap_nn(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, int(input_size*1.3)),
            
            nn.GELU(),
            nn.BatchNorm1d(int(input_size*1.3)),
            nn.Linear(int(input_size*1.3), int(input_size)),
            nn.Dropout(p=0.4),
            nn.GELU(),
            nn.BatchNorm1d(int(input_size)),
            nn.Linear(int(input_size), int(input_size*0.5)),
            nn.Dropout(p=0.4),
            nn.GELU(),
            nn.BatchNorm1d(int(input_size*0.5)),
            nn.Linear(int(input_size*0.5), output_size),
        )        
    def forward(self, x):
        # x = self.af(self.fc1(x))
        # x = self.b1(x)
        # x = self.af(self.fc2(x))
        # x = self.b2(x)
        # x = self.af(self.fc3(x))
        # x = self.b3(x)
        # x = self.af(self.fc4(x))
        
        return self.layers(x)
    
class Umap_data(Dataset):
    def __init__(self, embeddings, shrinking_factor):
        # timestamp = "1661853100"
        # with open(f'features_{timestamp}.npy', 'rb') as f:
            # self.features = np.load(f)
        self.features = embeddings
        reducer = umap.UMAP(
            n_components= int(embeddings.shape[1]*shrinking_factor)
        )

        self.targets = reducer.fit_transform(self.features)
        # total_no_samples = embeddings.shape[0]
        # val_idx = np.random.choice(np.arange(total_no_samples),size = int(total_no_samples*0.2), replace=False)
        
    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self, idx):
        input = torch.Tensor(self.features[idx,...])
        output = torch.Tensor(self.targets[idx,...])
        return input, output
    
def train_epoch(model, dataloader, loss_function, optimizer, device = 'cuda'):
    '''
    train a single epoch
    '''
    # sum of loss is initialized
    loss_sum = 0
    model.to(device)
    model.train()

    # for every single batch training process is done
    for (feature, target) in dataloader:
        feature = feature.float()
        target = target.float()
        samples_in_batch = target.shape[0]
        target_var = Variable(target).to(device) # .view((samples_in_batch, -1))
        #labels_var = labels.to(device)
        size_feature = feature.shape[1]
        feature_var = Variable(feature).to(device) # .view((samples_in_batch, size_feature))
        #images_var = images.to(device)
        # print(f'shape: {feature_var.shape} & {target_var.shape}')

        # Forward pass
        outputs = model(feature_var)

        # Backward and optimize
        model.zero_grad()
        optimizer.zero_grad()
        loss = loss_function(outputs, target_var)
        loss_sum += float(loss.item()) * target.size(0)/len(dataloader.dataset)

        loss.backward()
        optimizer.step()

    return loss_sum, model.eval()

def test_model(model, dataloader, loss_function, device = 'cuda'):
    '''
    validate/test a single epoch
    '''
    model.eval()
    with torch.no_grad():
        # sum of loss is initialized
        val_loss_sum = 0

        for (feature, target) in dataloader:

            target_var = Variable(target.float()).to(device)
            feature_var = Variable(feature.float()).to(device)

            # Forward pass
            outputs = model(feature_var)

            loss = loss_function(outputs, target_var)
            val_loss_sum += float(loss.item()) * target.size(0)/len(dataloader.dataset)

        return val_loss_sum
    
def train(model, train_loader, val_loader, optimizer, loss_function, epochs, device = 'cuda'):
    '''
    main method for training models
    '''
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    improvement_counter = 0
    best_val = 1.0
    best_model = None
    # training starts here
    best_after = int(0)
    print('training starts\n')
    for epoch in range(1, epochs+1):
        # train a single epoch
        train_loss, temp_model = train_epoch(model=model, dataloader=train_loader, loss_function=loss_function, optimizer=optimizer, device=device)
        # validate afterwards
        metrics = test_model(model=model, dataloader=val_loader, loss_function=loss_function, device=device)
        print(f'Epoch [{epoch}/{epochs}], Loss_train: {train_loss}, Loss_val: {metrics}')
        if metrics < best_val:
            best_val = metrics
            improvement_counter = 0
            best_after = epoch
            best_model = temp_model
            # torch.save(model.state_dict(), os.path.join(
            #     'optim_results', path_state_dict + ".pth"))
            # with open(os.path.join('optim_results', path_state_dict+".json"), 'w') as hyperparam_file:
            #     hyperparam_file.write(json.dumps(hyperparameters))
        else:
            improvement_counter += 1

        if improvement_counter >= 10:
            epochs_done = epoch
            break
        # append scores of epoch to arrays for plotting
        #loss_test_plot = np.append(loss_test_plot, val)
        #loss_plot = np.append(loss_plot, train)

    return best_val, best_after, epochs_done, best_model

def test_model(model, dataloader, loss_function, device = 'cuda'):
    '''
    validate/test a single epoch
    '''
    model.eval()
    with torch.no_grad():
        # sum of loss is initialized
        val_loss_sum = 0

        for (feature, target) in dataloader:

            target_var = Variable(target.float()).to(device)
            feature_var = Variable(feature.float()).to(device)

            # Forward pass
            outputs = model(feature_var)

            loss = loss_function(outputs, target_var)
            val_loss_sum += float(loss.item()) * target.size(0)/len(dataloader.dataset)

        return val_loss_sum
    

def get_nn_mapper(embeddings, shrinking_factor):
    net = Umuap_nn(input_size=embeddings.shape[1], output_size=int(embeddings.shape[1]*shrinking_factor))
    criterion = nn.MSELoss()
    EPOCHS = 200
    BATCH_SIZE = 16
    optm = Adam(net.parameters(), lr = 0.001, weight_decay=0.01)
    print('Fit UMAP ...')
    complete_ds = Umap_data(embeddings=embeddings, shrinking_factor=shrinking_factor)
    print('UMAP fitted, Data for Training Generated!')
    lengths = [int(len(complete_ds)*0.8), len(complete_ds) - int(len(complete_ds)*0.8)]
    train_ds, val_ds = torch.utils.data.random_split(complete_ds, lengths)
    val_loader = data.DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    train_loader = data.DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    _,_,_, nn_mapper = train(model=net, train_loader=train_loader, val_loader=val_loader, loss_function=criterion, optimizer=optm, epochs=EPOCHS)
    
    return nn_mapper
################################### UMAP MAPPER ################################################



class STPM(pl.LightningModule):
    def __init__(self, hparams):
        super(STPM, self).__init__()

        self.save_hyperparameters(hparams)
        self.dim_reducer = None
        self.measure_latences = False
        self.accelerator = "gpu"
        self.file_name_latences = None
        self.file_name_preparation_memory_bank = None
        # self.pooling = nn.AvgPool2d(kernel_size = args.avgpool_kernel, stride = args.avgpool_stride, padding = args.avgpool_padding)
        self.feature_maps_selected = args.feature_maps_selected
        self.pooling_strategy = args.pooling_strategy
        
        self.init_features()
        def hook_t(module, input, output): # hook func for extracting feature maps
            self.features.append(output)

        # 
        if args.backbone.__contains__('resnet'):
            self.model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True) # load pretrained weight
        elif args.backbone.__contains__('mobilenet'):
            self.model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        else:
            assert "No valid backbone"

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

        if args.backbone.__contains__('resnet'):
            if int(1) in self.feature_maps_selected:
                self.model.layer1[-1].register_forward_hook(hook_t) # using feature maps of layer 2&3
            if int(2) in self.feature_maps_selected:
                self.model.layer2[-1].register_forward_hook(hook_t) # using feature maps of layer 2&3
            if int(3) in self.feature_maps_selected:
                self.model.layer3[-1].register_forward_hook(hook_t)
            if int(4) in self.feature_maps_selected:
                self.model.layer4[-1].register_forward_hook(hook_t)
        elif args.backbone.__contains__('mobilenet'):
            for element in self.feature_maps_selected:
                self.model.features[element].register_forward_hook(hook_t)
        
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
        test_loader = DataLoader(test_datasets, batch_size=args.batch_size, shuffle=False, num_workers=0) #, pin_memory=True) # only work on batch_size=1, now.
        return test_loader

    def configure_optimizers(self):
        return None

    def on_train_start(self):
        self.model.eval() # to stop running_var move (maybe not critical)
        self.embedding_dir_path, self.sample_path, self.source_code_save_path = prep_dirs(args.project_root_path)
        self.embedding_list = []
    
    def on_test_start(self):
        if self.measure_latences:
            run_time = 0.0
            warm_up = 10
            reps = 5
            validate_cuda_measure = self.accelerator.__contains__('gpu')
            if validate_cuda_measure:
                run_time_validate = 0.0
            for _ in range(warm_up):
                _, _, _, = prep_dirs(args.project_root_path)
                _ = faiss.read_index(os.path.join(self.embedding_dir_path,'index.faiss'))
            for _ in range(reps):
                if self.accelerator.__contains__("gpu") and torch.cuda.is_available():
                    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True) # initialize cuda timers
                    starter.record() # start
                    st = time.perf_counter() # for validation
                elif self.accelerator.__contains__("cpu"):
                    st = time.perf_counter() # for devices not having cuda device
                else:
                    assert True, "Specified Accelerator not valid. Has to be \"cpu\" or \"gpu\"."
                ### process start ###
                self.embedding_dir_path, self.sample_path, self.source_code_save_path = prep_dirs(args.project_root_path)   
                self.index = faiss.read_index(os.path.join(self.embedding_dir_path,'index.faiss')) # load memory bank
                # if torch.cuda.is_available():
                #     res = faiss.StandardGpuResources()
                #     self.index = faiss.index_cpu_to_gpu(res, 0 ,self.index)
                self.init_results_list()
                ### process end ###
                if self.accelerator.__contains__("gpu") and torch.cuda.is_available():
                    et = time.perf_counter()
                    this_run_time = float((et - st) * 1e3)
                    ender.record()
                    torch.cuda.synchronize()
                    this_run_time_validate = starter.elapsed_time(ender)
                elif self.accelerator.__contains__("cpu"):
                    et = time.perf_counter()
                    this_run_time = float((et - st) * 1e3)
                run_time += this_run_time
                if validate_cuda_measure:
                    run_time_validate += this_run_time_validate  
            
            with open(self.file_name_preparation_memory_bank, 'w') as f:
                f.write(str(float(run_time / reps))) # mean
                
            if validate_cuda_measure:
                with open(self.file_name_preparation_memory_bank.split('.')[0] + '_validation.txt', 'w') as f2:
                    f2.write(str(float(run_time_validate / reps))) # mean  
        else:
            self.embedding_dir_path, self.sample_path, self.source_code_save_path = prep_dirs(args.project_root_path)   
            self.index = faiss.read_index(os.path.join(self.embedding_dir_path,'index.faiss')) # load memory bank
            # if torch.cuda.is_available():
            #     res = faiss.StandardGpuResources()
            #     self.index = faiss.index_cpu_to_gpu(res, 0 ,self.index)
            self.init_results_list()
            
    def embedding_concat_frame(self, embeddings):
        '''
        framework for concatenating more than two features or less than two
        '''
        no_of_embeddings = len(embeddings)
        if no_of_embeddings == int(1):
            embeddings_result = embeddings[0].cpu()
        elif no_of_embeddings == int(2):
            embeddings_result = embedding_concat(embeddings[0], embeddings[1])
        elif no_of_embeddings > int(2):
            for k in range(no_of_embeddings - 1):
                if k == int(0):
                    embeddings_result = embedding_concat(embeddings[0], embeddings[1]) # default
                    pass
                else:
                    if torch.cuda.is_available() and self.accelerator.__contains__("gpu"):
                        embeddings_result = embedding_concat(embeddings_result.cuda(), embeddings[k+1])
                    else:
                        embeddings_result = embedding_concat(embeddings_result, embeddings[k+1].cpu())
        return embeddings_result                
            
    def training_step(self, batch, batch_idx): # save locally aware patch features
        x, _, _, file_name, _ = batch
        features = self(x)
        embeddings = []
        for feature in features:
            pooled_feature = self.adaptive_pooling(feature)# using AvgPool2d to calculate local-aware features
            embeddings.append(pooled_feature)
        # embedding = embedding_concat(embeddings[0], embeddings[1])
        embedding = self.embedding_concat_frame(embeddings=embeddings) # shape (batch, 448, 16, 16) --> default
        self.embedding_list.extend(reshape_embedding(np.array(embedding)))

    def training_epoch_end(self, outputs): 
        total_embeddings = np.array(self.embedding_list) # shape: (115712, 448)
        ##### for dev #####
        # import time
        # with open(f'features_{int(time.time())}.npy', 'wb') as f:
        #     np.save(f, total_embeddings)
        ##### for dev #####
        # Random projection
        # self.randomprojector = SparseRandomProjection(n_components='auto', eps=0.9) # 'auto' => Johnson-Lindenstrauss lemma
        # fit
        if args.umap:
            print('\Fitting UMAP ...')
            shrinking_factor = args.shrinking_factor
            self.dim_reducer = umap.UMAP(n_components=int(total_embeddings.shape[1]*shrinking_factor))
            self.dim_reducer.fit(total_embeddings)
            total_embeddings_red = self.dim_reducer.transform(total_embeddings)
            print('Fitting Done!')
        else:
            total_embeddings_red = total_embeddings
            
        self.randomprojector = SparseRandomProjection(n_components=120, eps=0.9) # 'auto' => Johnson-Lindenstrauss lemma
        self.randomprojector.fit(total_embeddings_red)
        # Coreset Subsampling
        # selector = kCenterGreedyIden(total_embeddings,0,0)
        selector = kCenterGreedy(total_embeddings,0,0)
        
        selected_idx = selector.select_batch(model=self.randomprojector, already_selected=[], N=int(total_embeddings_red.shape[0]*float(args.coreset_sampling_ratio)))
        self.embedding_coreset = total_embeddings_red[selected_idx]
        
        print('initial embedding size : ', total_embeddings.shape)
        print('final embedding size : ', self.embedding_coreset.shape)
        #faiss
        self.index = faiss.IndexFlatL2(self.embedding_coreset.shape[1]) # original self.embedding_coreset.shape[1] = dimension
        self.index.add(self.embedding_coreset) 
        faiss.write_index(self.index,  os.path.join(self.embedding_dir_path,'index.faiss'))
    
    def adaptive_pooling(self, feature):
        if self.pooling_strategy == '':
            m = nn.AvgPool2d(kernel_size = 3, stride = 1, padding = 1)
        
        if self.pooling_strategy == '1':
            # layer 1 & 3
            if feature.shape[3] == 16:
                m = nn.AvgPool2d(kernel_size = 3, stride = 2, padding = 1)
            else:
                m = nn.AvgPool2d(kernel_size = 3, stride = 1, padding = 1)
        if self.pooling_strategy == '1.1':
            # layer 1 & 3
            if feature.shape[3] == 16:
                m = nn.AvgPool2d(kernel_size = 7, stride = 3, padding = 1)
            else:
                m = nn.AvgPool2d(kernel_size = 3, stride = 1, padding = 1)
        if self.pooling_strategy == '1.2':
            # layer 1 & 3
            if feature.shape[3] == 16:
                m = nn.AvgPool2d(kernel_size = 7, stride = 3, padding = 1)
            elif feature.shape[3] == 8:
                m = nn.AvgPool3d(kernel_size = 3, stride = (1,2,2), padding = 1)
            else:
                m = nn.AvgPool3d(kernel_size = 3, stride = (2,1,1), padding = 1)
        if self.pooling_strategy == '2':
            # layer 1 & 3
            if feature.shape[3] == 16:
                m = nn.MaxPool2d(kernel_size = 5, stride = 2, padding = 1)
            else:
                m = nn.MaxPool2d(kernel_size = 5, stride = 1, padding = 1)
        if self.pooling_strategy == '3':
        # layer 1 & 3
            if feature.shape[3] == 16:
                m = nn.AvgPool2d(kernel_size = 5, stride = 2, padding = 1)
            else:
                m = nn.AvgPool2d(kernel_size = 5, stride = 1, padding = 1)
        
        if self.pooling_strategy == '4':
        # layer 1 & 3
            if feature.shape[3] == 16:
                m = nn.MaxPool2d(kernel_size = 5, stride = 2, padding = 1)
            else:
                m = nn.MaxPool2d(kernel_size = 5, stride = 1, padding = 1)
        if self.pooling_strategy == 'adapt_avg_4':
            m = nn.AdaptiveAvgPool2d(4)
        if self.pooling_strategy == 'adapt_avg_6':
            m = nn.AdaptiveAvgPool2d(6)
        if self.pooling_strategy == 'adapt_avg_8':
            m = nn.AdaptiveAvgPool2d(8)    
        if self.pooling_strategy == 'adapt_avg_10':
            m = nn.AdaptiveAvgPool2d(10)
        if self.pooling_strategy == 'adapt_avg_12':
            m = nn.AdaptiveAvgPool2d(12)
        if self.pooling_strategy == 'adapt_avg_14':
            m = nn.AdaptiveAvgPool2d(14)
        if self.pooling_strategy == 'adapt_avg_16':
            m = nn.AdaptiveAvgPool2d(16)
        if self.pooling_strategy == 'adapt_max_4':
            m = nn.AdaptiveMaxPool2d(4)
        if self.pooling_strategy == 'adapt_max_6':
            m = nn.AdaptiveMaxPool2d(6)
        if self.pooling_strategy == 'adapt_max_8':
            m = nn.AdaptiveMaxPool2d(8)    
        if self.pooling_strategy == 'adapt_max_10':
            m = nn.AdaptiveMaxPool2d(10)
        if self.pooling_strategy == 'adapt_max_12':
            m = nn.AdaptiveMaxPool2d(12)
        if self.pooling_strategy == 'adapt_max_14':
            m = nn.AdaptiveMaxPool2d(14)
        if self.pooling_strategy == 'adapt_max_16':
            m = nn.AdaptiveMaxPool2d(16)
        return m(feature)
    
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
            feature_pooled = self.adaptive_pooling(feature) # define avg Pooling filter
            embeddings.append(feature_pooled) # add to embeddings pooled feature maps, does not change shape
        # embedding_ = embedding_concat(embeddings[0], embeddings[1]) # concat two feature maps using unfold() from torch, leads to torch with shape [1, 384, 8, 8]
        embedding_ = self.embedding_concat_frame(embeddings)
        if x.size()[0] <= 1:
            embedding_test = np.array(reshape_embedding(np.array(embedding_))) # reshape the features as 1-D vectors, save them as numpy ndarray, shape: [64,384] for batch_size = 1
            # resulting_features = embedding_test.shape[0]
            if args.umap:
                embedding_ = self.dim_reducer.transform(embedding_)
            score_patches, _ = self.index.search(embedding_test , k=args.n_neighbors) # brutal force search of k nearest neighbours using faiss.IndexFlatL2.search; shape [64,9], memory bank is utilizied     
            anomaly_map = score_patches[:,0].reshape((int(math.sqrt(len(score_patches[:,0]))),int(math.sqrt(len(score_patches[:,0])))))
            a = int(args.load_size) # int, 64 
            anomaly_map_resized = cv2.resize(anomaly_map, (a, a)) # [8,8] --> [64,64]
            anomaly_map_resized_blur = gaussian_filter(anomaly_map_resized, sigma=4)# shape [8,8]
        else:
            embedding_test = np.array([np.array(reshape_embedding(np.array(embedding_[k,...].unsqueeze(0)))) for k in range(x.size()[0])])
            # resulting_features = embedding_test[0].shape[0]
            if args.umap:
                print('\nTransform Features started...')
                temp_flatten = np.reshape(embedding_test,(embedding_test.shape[0]*embedding_test.shape[1], embedding_test.shape[2]))
                transformed_flatten = self.dim_reducer.transform(temp_flatten)
                embedding_test = np.reshape(transformed_flatten,(embedding_test.shape[0], embedding_test.shape[1], int(embedding_test.shape[2]*args.shrinking_factor)))
                # embedding_test = [self.dim_reducer.transform(element) for element in embedding_test] # not working properly, extreme long inference time
                print('... finished!')
            score_patches = [self.index.search(this_embedding_test, k=args.n_neighbors)[0] for this_embedding_test in embedding_test]
            anomaly_map = [score_patch[:,0].reshape((int(math.sqrt(len(score_patch[:,0]))),int(math.sqrt(len(score_patch[:,0]))))) for score_patch in score_patches]
            a = int(args.load_size)
            anomaly_map_resized = [cv2.resize(this_anomaly_map, (a, a)) for this_anomaly_map in anomaly_map]
            anomaly_map_resized_blur = [gaussian_filter(this_anomaly_map_resized, sigma=4) for this_anomaly_map_resized in anomaly_map_resized]
           
        return score_patches, anomaly_map_resized_blur
    
    def prediction_process_core_piecwise_measuremet(self, batch, batch_idx):
        '''
        Extracted core process of prediction for better readability.
        '''
        if torch.cuda.is_available():
            t_0_gpu, t_1_gpu, t_1_1_gpu, t_2_gpu, t_3_gpu, t_4_gpu = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        # input // start
        ######################################################################################################################################################
        # pass data through CNN // feature extraction
        t_0_cpu = time.perf_counter()
        if torch.cuda.is_available() and self.accelerator.__contains__("gpu"): 
            t_0_gpu.record()
            torch.cuda.synchronize()
        x, gt, label, file_name, x_type = batch # x: feature/input; gt: mask for pixel wise classification ground truth 
        # extract embedding        
        features = self(x)
        ######################################################################################################################################################
        # embedding of features
        t_1_cpu = time.perf_counter()
        if torch.cuda.is_available() and self.accelerator.__contains__("gpu"):
            t_1_gpu.record()
            torch.cuda.synchronize()
        embeddings = []
        for feature in features: # features: list of feature maps, size: [1,128,8,8] & [1,256,4,4] (first entry --> batch_size = 1)
            feature_pooled = self.adaptive_pooling(feature) # define avg Pooling filter
            embeddings.append(feature_pooled) # add to embeddings pooled feature maps, does not change shape
        # embedding_ = embedding_concat(embeddings[0], embeddings[1]) # concat two feature maps using unfold() from torch, leads to torch with shape [1, 384, 8, 8]
        embedding_ = self.embedding_concat_frame(embeddings)
        if x.size()[0] <= 1:
            embedding_test = np.array(reshape_embedding(np.array(embedding_))) # reshape the features as 1-D vectors, save them as numpy ndarray, shape: [64,384] for batch_size = 1
            t_1_1_cpu = None
            if args.umap:
                t_1_1_cpu = time.perf_counter()
                if torch.cuda.is_available() and self.accelerator.__contains__("gpu"):
                    t_1_1_gpu.record()
                    torch.cuda.synchronize()
                print('\nTransform Features started...')
                temp_flatten = np.reshape(embedding_test,(embedding_test.shape[0]*embedding_test.shape[1], embedding_test.shape[2]))
                transformed_flatten = self.dim_reducer.transform(temp_flatten)
                embedding_test = np.reshape(transformed_flatten,(embedding_test.shape[0], embedding_test.shape[1], int(embedding_test.shape[2]*args.shrinking_factor)))
                # embedding_test = [self.dim_reducer.transform(element) for element in embedding_test] # not working properly, extreme long inference time
                print('... finished!')
            resulting_features = (embedding_test.shape[0], embedding_test.shape[1]) 
        ######################################################################################################################################################
            # comparison with memory bank
            t_2_cpu = time.perf_counter() 
            if torch.cuda.is_available() and self.accelerator.__contains__("gpu"):
                t_2_gpu.record()
                torch.cuda.synchronize()
            score_patches, _ = self.index.search(embedding_test , k=args.n_neighbors) # brutal force search of k nearest neighbours using faiss.IndexFlatL2.search; shape [64,9], memory bank is utilizied
        ######################################################################################################################################################
            # create anomaly map
            t_3_cpu = time.perf_counter() 
            if torch.cuda.is_available() and self.accelerator.__contains__("gpu"):
                t_3_gpu.record()
                torch.cuda.synchronize()
            anomaly_map = score_patches[:,0].reshape((int(math.sqrt(len(score_patches[:,0]))),int(math.sqrt(len(score_patches[:,0])))))
            a = int(args.load_size) # int, 64 
            anomaly_map_resized = cv2.resize(anomaly_map, (a, a)) # [8,8] --> [64,64]
            anomaly_map_resized_blur = gaussian_filter(anomaly_map_resized, sigma=4)# shape [8,8]
        else:
        ######################################################################################################################################################
            # comparison with memory bank
            embedding_test = np.array([np.array(reshape_embedding(np.array(embedding_[k,...].unsqueeze(0)))) for k in range(x.size()[0])])
            t_1_1_cpu = None
            if args.umap:
                t_1_1_cpu = time.perf_counter()
                if torch.cuda.is_available() and self.accelerator.__contains__("gpu"):
                    t_1_1_gpu.record()
                    torch.cuda.synchronize()

                    temp_flatten = np.reshape(embedding_test,(embedding_test.shape[0]*embedding_test.shape[1], embedding_test.shape[2]))
                    transformed_flatten = self.dim_reducer.transform(temp_flatten)
                    embedding_test = np.reshape(transformed_flatten,(embedding_test.shape[0], embedding_test.shape[1], int(embedding_test.shape[2]*args.shrinking_factor)))
            # embedding_test = [np.array(reshape_embedding(np.array(embedding_[k,...].unsqueeze(0)))) for k in range(x.size()[0])]
            t_2_cpu = time.perf_counter() 
            if torch.cuda.is_available() and self.accelerator.__contains__("gpu"):
                t_2_gpu.record()
                torch.cuda.synchronize()
            
            resulting_features = (embedding_test[0].shape[0], embedding_test[0].shape[1])
        ######################################################################################################################################################
            # create anomaly map
            t_3_cpu = time.perf_counter() 
            if torch.cuda.is_available() and self.accelerator.__contains__("gpu"):
                t_3_gpu.record() 
                torch.cuda.synchronize()
            score_patches = [self.index.search(this_embedding_test, k=args.n_neighbors)[0] for this_embedding_test in embedding_test]
            anomaly_map = [score_patch[:,0].reshape((int(math.sqrt(len(score_patch[:,0]))),int(math.sqrt(len(score_patch[:,0]))))) for score_patch in score_patches]
            a = int(args.load_size)
            anomaly_map_resized = [cv2.resize(this_anomaly_map, (a, a)) for this_anomaly_map in anomaly_map]
            anomaly_map_resized_blur = [gaussian_filter(this_anomaly_map_resized, sigma=4) for this_anomaly_map_resized in anomaly_map_resized]
        ###################################################################################################################################################### 
        # end
        t_4_cpu = time.perf_counter() 
        if torch.cuda.is_available() and self.accelerator.__contains__("gpu"):
            t_4_gpu.record()
            torch.cuda.synchronize()
        else:
            t_0_gpu, t_1_gpu, t_2_gpu, t_3_gpu, t_4_gpu, t_1_1_gpu = None, None, None, None, None, None
        return score_patches, anomaly_map_resized_blur, resulting_features, t_0_cpu, t_1_cpu, t_2_cpu, t_3_cpu, t_4_cpu, t_0_gpu, t_1_gpu, t_2_gpu, t_3_gpu, t_4_gpu, t_1_1_cpu, t_1_1_gpu
    
    def eval_one_step_test(self, score_patches, anomaly_map_resized_blur, x, gt, label, file_name, x_type):
        '''
        Extracted evaluation of single output
        '''
        if x.dim() != 4:
            x, gt, label = x.unsqueeze(0), gt.unsqueeze(0), label.unsqueeze(0)
        N_b = score_patches[np.argmax(score_patches[:,0])].astype(np.longfloat) # max of each patch
        N_b_exp = np.exp(N_b)
        N_b_exp = N_b_exp[~np.isinf(N_b_exp)]
        if len(N_b_exp) > 0:
            w = (1 - (np.max(N_b_exp))/np.sum(N_b_exp)) # scaling factor in paper
        else:
            w = np.nan
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

    
    def test_step(self, batch, batch_idx): # Nearest Neighbour Search
        
        if self.measure_latences:
            # print(f'CUDA AVAILABLE? {torch.cuda.is_available()}\n')
            validate_cuda_measure = self.accelerator.__contains__('gpu') # cause this makes only sense with accelerator gpu
            warm_up = 1 # specify how often file should be processed before actual measurment
            reps = 3 # repititions for more meaningful measurements due to averaging
            # run_times = [] # initialize timer
            # run_times_validate = []
            run_times = {
                    '#1 feature extraction cpu': [],
                    '#2 feature extraction gpu': [],
                    '#3 embedding of features cpu': [],
                    '#4 embedding of features gpu': [],
                    '#5 search with memory bank cpu': [],
                    '#6 search with memory bank gpu': [],
                    '#7 anomaly map cpu': [],
                    '#8 anomaly map gpu': [],
                    '#9 sum cpu': [],
                    '#10 sum gpu': [],
                    '#11 whole process cpu': [],
                    '#12 whole process gpu': [],
                    '#15 dim reduction cpu': [],
                    '#14 dim reduction gpu': []          
                }
            # self.file_name_latences = 'latences.txt'
            # warm_up
            for _ in range(warm_up):
                _, _ = self.prediction_process_core(batch, batch_idx)
            # measurement
            for _ in range(reps):
                #start timer
                if self.accelerator.__contains__("gpu") and torch.cuda.is_available():
                    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True) # initialize cuda timers
                    starter.record() # start
                st = time.perf_counter() # for devices not having a cuda device
                all_score_patches, all_anomaly_map_resized_blur, resulting_features, t_0_cpu, t_1_cpu, t_2_cpu, t_3_cpu, t_4_cpu, t_0_gpu, t_1_gpu, t_2_gpu, t_3_gpu, t_4_gpu, t_1_1_cpu, t_1_1_gpu = self.prediction_process_core_piecwise_measuremet(batch, batch_idx) # core process
                et = time.perf_counter()
                if self.accelerator.__contains__("gpu") and torch.cuda.is_available(): # in case gpu is used
                    ender.record() #end
                    torch.cuda.synchronize() # wait for gpu sync
                    run_times['#12 whole process gpu'] += [starter.elapsed_time(ender)] # in ms, run time of process
                    run_times['#2 feature extraction gpu'] += [t_0_gpu.elapsed_time(t_1_gpu)]
                    run_times['#4 embedding of features gpu'] += [t_1_gpu.elapsed_time(t_2_gpu)]
                    run_times['#6 search with memory bank gpu'] += [t_2_gpu.elapsed_time(t_3_gpu)]
                    run_times['#8 anomaly map gpu'] += [t_3_gpu.elapsed_time(t_4_gpu)]
                    run_times['#10 sum gpu'] += [t_0_gpu.elapsed_time(t_4_gpu)]
                    run_times['#14 dim reduction gpu'] += [t_1_1_gpu.elapsed_time(t_2_gpu)] 
                else: # in case cpu is used
                    run_times['#12 whole process gpu'] += [0.0] # in ms, run time of process
                    run_times['#2 feature extraction gpu'] += [0.0]
                    run_times['#4 embedding of features gpu'] += [0.0]
                    run_times['#6 search with memory bank gpu'] += [0.0]
                    run_times['#8 anomaly map gpu'] += [0.0]
                    run_times['#10 sum gpu'] += [0.0]
                    run_times['#14 dim reduction gpu'] += [0.0]
                run_times['#1 feature extraction cpu'] += [float((t_1_cpu - t_0_cpu) * 1e3)]
                run_times['#3 embedding of features cpu'] += [float((t_2_cpu - t_1_cpu) * 1e3)]
                run_times['#5 search with memory bank cpu'] += [float((t_3_cpu - t_2_cpu) * 1e3)]
                run_times['#7 anomaly map cpu'] += [float((t_4_cpu - t_3_cpu) * 1e3)]
                run_times['#9 sum cpu'] += [float((t_4_cpu - t_0_cpu) * 1e3)]
                run_times['#11 whole process cpu'] += [float((et - st) * 1e3)]
                run_times['#15 dim reduction cpu'] += [float((t_2_cpu - t_1_1_cpu) * 1e3)]
            assert len(run_times['#1 feature extraction cpu']) == reps, "Something went wrong!"
            for this_entry in run_times.items():
                run_times[this_entry[0]] = float((sum(this_entry[1]) / len(this_entry[1])) / batch[0].size()[0]) # mean
            run_times['#13 preparation memory bank'] = float(np.genfromtxt(model.file_name_preparation_memory_bank))
            run_times['batch_size'] = args.__dict__['batch_size']
            run_times['input_size'] = args.__dict__['input_size']
            run_times['coreset_sampling_ratio'] = args.__dict__['coreset_sampling_ratio']
            run_times['n_neighbors'] = args.__dict__['n_neighbors']
            run_times['pooling_strategy'] = args.__dict__['pooling_strategy']
            run_times['resulting_features_spatial'] = resulting_features[0]
            run_times['resulting_features_depth'] = resulting_features[1]
            run_times['backbone'] = args.__dict__['backbone']
            # if validate_cuda_measure:
                # print(run_times)
            pd_run_times = pd.DataFrame(run_times, index=[batch_idx])
            if os.path.exists(os.path.join(os.path.dirname(__file__), "results","csv", self.file_name_latences)):
                pd_run_times_ = pd.read_csv(os.path.join(os.path.dirname(__file__), "results", "csv",self.file_name_latences), index_col=0)
                pd_run_times = pd.concat([pd_run_times_, pd_run_times], axis=0)
                pd_run_times.to_csv(os.path.join(os.path.dirname(__file__), "results","csv", self.file_name_latences))
            else:
                pd_run_times.to_csv(os.path.join(os.path.dirname(__file__), "results","csv",self.file_name_latences))    
        else:
            all_score_patches, all_anomaly_map_resized_blur = self.prediction_process_core(batch, batch_idx) # core process
        # calculating of scores and saving of results
        if type(all_score_patches) == list:
            results = (all_score_patches, all_anomaly_map_resized_blur)
            x_batch, gt_batch, label_batch, file_name_batch, x_type_batch = batch
            for k in range(x_batch.size()[0]):
                score_patches, anomaly_map_resized_blur = results[0][k], results[1][k]
                x, gt, label, file_name, x_type = x_batch[k], gt_batch[k], label_batch[k], file_name_batch[k], x_type_batch[k]
                self.eval_one_step_test(score_patches, anomaly_map_resized_blur, x, gt, label, file_name, x_type)
        else:
            score_patches, anomaly_map_resized_blur = all_score_patches, all_anomaly_map_resized_blur
            x, gt, label, file_name, x_type = batch
            self.eval_one_step_test(score_patches, anomaly_map_resized_blur, x, gt, label, file_name, x_type)
                

    def test_epoch_end(self, outputs):
        # print("Total pixel-level auc-roc score :")
        # print(self.gt_list_img_lvl)
        # print(self.pred_list_img_lvl)
        pixel_auc = roc_auc_score(self.gt_list_px_lvl, self.pred_list_px_lvl)
        # print(pixel_auc)
        # print("Total image-level auc-roc score :")
        img_auc = roc_auc_score(self.gt_list_img_lvl, self.pred_list_img_lvl)
        # print(img_auc)
        # print('test_epoch_end')
        values = {'pixel_auc': pixel_auc, 'img_auc': img_auc}
        self.log_dict(values)
        anomaly_list = []
        normal_list = []
        for i in range(len(self.gt_list_img_lvl)):
            if self.gt_list_img_lvl[i] == 1:
                anomaly_list.append(self.pred_list_img_lvl[i])
            else:
                normal_list.append(self.pred_list_img_lvl[i])
        
        if os.path.exists(os.path.join(os.path.dirname(__file__), "results","csv", self.file_name_latences)) and self.measure_latences:
            pd_run_times_ = pd.read_csv(os.path.join(os.path.dirname(__file__), "results", "csv",self.file_name_latences), index_col=0)
            pd_results = pd.DataFrame({'img_auc': [img_auc]*pd_run_times_.shape[0], 'pixel_auc': [pixel_auc]*pd_run_times_.shape[0]})
            pd_run_times = pd.concat([pd_run_times_, pd_results], axis=1)
            pd_run_times.to_csv(os.path.join(os.path.dirname(__file__), "results", "csv",self.file_name_latences))
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
    parser.add_argument('--num_epochs', default=1, type = int) # 1 iteration is enough
    parser.add_argument('--batch_size', default=56, type = int)
    parser.add_argument('--load_size', default=64, type = int) 
    parser.add_argument('--input_size', default=64, type = int) # using same input size and load size for our data
    parser.add_argument('--feature_maps_selected', default=[3], type=int, nargs='+')
    parser.add_argument('--coreset_sampling_ratio', default=0.01, type = float)
    parser.add_argument('--pooling_strategy', default='', type = str)
    parser.add_argument('--avgpool_kernel', default=3, type=int)
    parser.add_argument('--avgpool_stride', default=1, type=int)
    parser.add_argument('--avgpool_padding', default=1, type=int)
    parser.add_argument('--backbone', default = 'resnet', type = str)
    parser.add_argument('--project_root_path', default=r'./test') # location to save result
    parser.add_argument('--save_src_code', default=True) 
    parser.add_argument('--save_anomaly_map', default=True)
    parser.add_argument('--n_neighbors', type=int, default=9)
    parser.add_argument('--propotion', type=int, default=452) # number of training samples used, default 452=all samples 
    parser.add_argument('--pre_weight', default='imagenet')
    parser.add_argument('--file_name_latences', default='latences.csv', type = str)
    parser.add_argument('--accelerator', default=None)
    parser.add_argument('--devices', default=None)
    parser.add_argument('--umap', type=bool, default=True)
    parser.add_argument('--shrinking_factor', type=float, default=0.25)
    # editable
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # not needed anywhere
    accelerator = Acceler(1).name # choose 1 for gpu, 2 for cpu
    if not os.path.exists(os.path.join(os.path.dirname(__file__), "results", "csv")):
        os.makedirs(os.path.join(os.path.dirname(__file__), "results","csv"))
    if not os.path.exists(os.path.join(os.path.dirname(__file__), "results", "temp")):
        os.makedirs(os.path.join(os.path.dirname(__file__), "results","temp"))
    args = get_args()
    trainer = pl.Trainer.from_argparse_args(args, default_root_dir=os.path.join(args.project_root_path, args.category), max_epochs=args.num_epochs, accelerator=accelerator) #, gpus=1) #, check_val_every_n_epoch=args.val_freq,  num_sanity_val_steps=0) # ,fast_dev_run=True)
    model = STPM(hparams=args)
    model.measure_latences = True # if not specified this is False
    model.file_name_latences = args.__dict__['file_name_latences']
    model.file_name_preparation_memory_bank = os.path.join(os.path.dirname(__file__), "results", "temp", args.__dict__['file_name_latences'].split('.')[0] + '_preparation_memory_bank.csv')
    model.accelerator = accelerator
    if args.phase == 'train':
        trainer.fit(model)
        trainer.test(model)
    elif args.phase == 'test':
        trainer.test(model)
        
    # all_latencies = genfromtxt(model.file_name_latences, delimiter='\t')[:-1]
    # print(f'Average Latency: {round(sum(all_latencies)/len(all_latencies), 3)} ms')
    
    # if accelerator.__contains__('gpu'):
    #     all_latencies_validate = genfromtxt(model.file_name_latences.split('.')[0] + '_validation.txt', delimiter='\t')[:-1]
    #     print(f'Average Latency Validation: {round(sum(all_latencies_validate)/len(all_latencies_validate), 3)} ms')

    # preparation_memory_bank = float(genfromtxt(model.file_name_preparation_memory_bank))
    # print(f'Preparation of Memory Bank: {round(preparation_memory_bank, 3)} ms')

    # if accelerator.__contains__('gpu'):
    #     preparation_memory_bank = float(genfromtxt(model.file_name_preparation_memory_bank.split('.')[0] + '_validation.txt'))
    #     print(f'Preparation of Memory Bank Validation: {round(preparation_memory_bank, 3)} ms')
    # DONE
    # annotations to code for better understanding (!)
    # changed order of some lines because some are not part of prediction process but of scoring process and therefore not relevant for run time
    # generate func for core process for better overview
    # warm_up loop
    # measurement loop
    # different measurements for GPU and CPU 
    # added args to model:
        # - wether measurement should be done
        # - file names for latences and memory bank
    # latences and preparation duration for memory bank are written into csv file
    # print of measurements in main part
    # added arg 'accelerator' to compare runtime of cpu and gpu

    # TODO
    # determine final baseline
    # batch-size --> try different batch sizes! --> validate
    # different variants
    # debug cpu runtime memory bank
    # sample ratio testenFwar
    # portions of each step for latences
    # Anaysis: input_size, sample_ratio, batch_size, k of nn search, patch_size/ stride of m Avg_pool stride = (2,2) maybe (3,3)? --> csv 
    # faiss-gpu bei Zeit