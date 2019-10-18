import os 
import numpy as np 
import pandas as pd
import torch  
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from skimage import io
import albumentations as albu
from albumentations.pytorch.transforms import ToTensor

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop([224, 224]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
def get_transforms(phase, width=224, height=224):
    list_transforms = []
    if phase == "train":
        list_transforms.extend(
            [
            albu.VerticalFlip(p=0.5),
            albu.RandomRotate90(p=0.5),
            albu.OneOf([
                albu.RandomContrast(),
                albu.RandomGamma(),
                albu.RandomBrightness(),
            ], p=0.3),
            albu.OneOf([
                albu.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                albu.GridDistortion(p=0.5),
                albu.OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)                  
            ], p=0.5),
            albu.ShiftScaleRotate(),
            ]
        )
    list_transforms.extend(
        [
            albu.Resize(width,height,always_apply=True),
            # albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
            ToTensor(),
        ]
    )
    list_trfms = albu.Compose(list_transforms)
    return list_trfms
class vinDataset(Dataset):
    def __init__(self, root_dir, file_name, num_triplet, phase):
        super(vinDataset, self).__init__()
        self.root_dir = root_dir 
        self.file_name = file_name 
        self.num_triplet = num_triplet
        self.transforms = get_transforms(phase, width = 224, height = 224)
        self.phase = phase 
        self.df = self.__read_data__(root_dir=root_dir, file_name = file_name)
        self.training_triplets = self.generate_triplets(self.df, self.num_triplet)

    def __getitem__(self, idx):            
        anc_id, pos_id, neg_id, pos_class, neg_class, anc_name, pos_name, neg_name = self.training_triplets[idx]      
        anc_img   = os.path.join(self.root_dir, anc_name) 
        pos_img   = os.path.join(self.root_dir, pos_name) 
        neg_img   = os.path.join(self.root_dir, neg_name) 
        
        anc_img   = io.imread(anc_img)
        pos_img   = io.imread(pos_img)
        neg_img   = io.imread(neg_img)

        pos_class = torch.from_numpy(np.array([pos_class]).astype('long'))
        neg_class = torch.from_numpy(np.array([neg_class]).astype('long'))
        
        sample = {'anc_img': anc_img, 'pos_img': pos_img, 'neg_img': neg_img, 'pos_class': pos_class, 'neg_class': neg_class}

        # dict_sample = {'anc_img': anc_img, 'pos_img': pos_img, 'neg_img': neg_img}
        augmented_anc = self.transforms(image=anc_img)
        augmented_pos = self.transforms(image=pos_img)
        augmented_neg = self.transforms(image=neg_img)
        sample['anc_img'] = augmented_anc['image']
        sample['pos_img'] = augmented_pos['image']
        sample['neg_img'] = augmented_neg['image']
            
        return sample

    def __len__(self):
        return len(self.training_triplets)
    
    def __read_data__(self, root_dir, file_name):
        root_dir = '/'.join(root_dir.split('/')[:-3])

        df = pd.read_csv(os.path.join(root_dir, file_name))
        df = df.reset_index(drop=True)
        return df 
    
    @staticmethod
    def generate_triplets(df, num_triplets):
        
        def make_dictionary_for_face_class(df):

            '''
              - face_classes = {'class0': [class0_id0, ...], 'class1': [class1_id0, ...], ...}
            '''
            face_classes = dict()
            for idx, label in enumerate(df['id']):
                if label not in face_classes:
                    face_classes[label] = []
                face_classes[label].append([df.loc[idx, 'id'], df.loc[idx, 'imageName']])
            return face_classes
        
        triplets    = []
        classes     = df['id'].unique()
        face_classes = make_dictionary_for_face_class(df)
        for _ in range(num_triplets):

            '''
              - randomly choose anchor, positive and negative images for triplet loss
              - anchor and positive images in pos_class
              - negative image in neg_class
              - at least, two images needed for anchor and positive images in pos_class
              - negative image should have different class as anchor and positive images by definition
            '''
        
            pos_class = np.random.choice(classes)
            neg_class = np.random.choice(classes)
            while len(face_classes[pos_class]) < 2:
                pos_class = np.random.choice(classes)
            while pos_class == neg_class:
                neg_class = np.random.choice(classes)
            
            # pos_name = df.loc[df['id'] == pos_class, 'name'].values[0]
            # neg_name = df.loc[df['id'] == neg_class, 'name'].values[0]

            if len(face_classes[pos_class]) == 2:
                ianc, ipos = np.random.choice(2, size = 2, replace = False)
            else:
                ianc = np.random.randint(0, len(face_classes[pos_class]))
                ipos = np.random.randint(0, len(face_classes[pos_class]))
                while ianc == ipos:
                    ipos = np.random.randint(0, len(face_classes[pos_class]))
            ineg = np.random.randint(0, len(face_classes[neg_class]))
            
            triplets.append([face_classes[pos_class][ianc][0], face_classes[pos_class][ipos][0], face_classes[neg_class][ineg][0], 
                             pos_class, neg_class, face_classes[pos_class][ianc][1], face_classes[pos_class][ipos][1], face_classes[neg_class][ineg][1]])
        
        return triplets

if __name__ == "__main__":
    # df = pd.read_csv('/home/toandm2/code/DATASETS/Hands/HandInfo.csv')
    # df = df.reset_index(drop=True)
    dataset = vinDataset(root_dir='/home/toandm2/code/DATASETS/Hands/', file_name='HandInfo.csv', num_triplet = 10, phase='train')
    print(dataset[0])
    # img, label = dataset[0]
    # print(img)
    # print(label)
    pass 
