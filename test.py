import os 
import numpy as np 
import pandas as pd 
import cv2
import torch 
from torchvision.transforms import Resize
from models import FaceNetModel
from datasets import get_transforms
import time 
from sklearn import neighbors
from sklearn.svm import SVC
from glob import glob 
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class Recognition(object):
    def __init__(self, weights):
        super(Recognition, self).__init__()
        torch.backends.cudnn.benchmark=True
        self.weights = weights 
        self.model = FaceNetModel(embedding_size=128, num_classes=10000)
        self.device, device_ids = self._prepare_device([0])
        # self.device = torch.device('cpu')
        self.model = self.model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
        self.transforms = get_transforms(phase='valid', width = 224, height = 224)
        if self.weights is not None:
            print('Load Checkpoint')
            checkpoint = torch.load(weights, map_location=lambda storage, loc: storage)
            self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        # init
        self.image = torch.FloatTensor(1, 3, 224, 224)
        self.image = self.image.to(self.device)
    def process(self, images):
        cpu_image = self.transforms(image=images)['image'].unsqueeze(0)
        self.loadData(self.image, cpu_image)
        output = self.model(self.image)
        return output
    
    def train_cls(self):
        knn_clf = neighbors.KNeighborsClassifier(n_neighbors=3, weights='distance')
        clf = SVC(gamma='auto')
        df = pd.read_csv('./data/HandInfo.csv')
        df = df.reset_index(drop=True)
        X = []
        y = []
        train_df, val_df = train_test_split(df, test_size=0.1, stratify=df['id'])
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        for idx in tqdm(range(len(train_df))):
            path = os.path.join('./data/Hands', train_df.loc[idx, 'imageName'])
            feature = np.squeeze(self.process(cv2.imread(path)).cpu().data.numpy())
            X.append(feature)
            y.append(train_df.loc[idx, 'id'])
        # knn_clf.fit(X, y)
        clf.fit(X, y)
        correct = 0
        total = 0
        for idx in tqdm(range(len(val_df))):
            path = os.path.join('./data/Hands', val_df.loc[idx, 'imageName'])
            feature = self.process(cv2.imread(path)).cpu().data.numpy()
            pred = clf.predict(feature)
            lab = val_df.loc[idx, 'id']
            if pred[0]==lab:
                correct +=1
            total +=1
        
        print(correct*100/total)
    
    @staticmethod
    def loadData(v, data):
        with torch.no_grad():
            v.resize_(data.size()).copy_(data)
    @staticmethod
    def _prepare_device(device):
        if type(device)==int:
            n_gpu_use = device
        else:
            n_gpu_use = len(device)
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print("Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        if type(device)==int:
            device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
            list_ids = list(range(n_gpu_use))
        elif len(device) == 1:
            list_ids = device
            if device[0] >= 0 and device[0] < n_gpu:    
                device = torch.device('cuda:{}'.format(device[0]))
            else:
                device = torch.device('cpu')
        else:
            list_ids = device
            device = torch.device('cuda:{}'.format(device[0]) if n_gpu_use > 0 else 'cpu')
            
        return device, list_ids
        
if __name__=='__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
    reg = Recognition(weights='/root/code/Hand-Recognition/saved/FaceNetModel/model_best.pth')
    reg.train_cls()
