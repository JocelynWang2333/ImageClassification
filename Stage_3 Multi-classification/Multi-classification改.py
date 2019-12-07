import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
from Multi_Network import *
from torchvision.transforms import transforms
from PIL import Image
import pandas as pd
import random
from torch import optim
from torch.optim import lr_scheduler
import copy

ROOT_DIR = 'Dataset/'
TRAIN_DIR = 'train/'
VAL_DIR = 'val/'
TRAIN_ANNO = 'Multi_train_annotation.csv'
VAL_ANNO = 'Multi_val_annotation.csv'
CLASSES = ['Mammals', 'Birds']
SPECIES = ['rabbits', 'rats', 'chickens']

class MyDataset():

    def __init__(self, root_dir, annotations_file, transform=None):

        self.root_dir = root_dir
        self.annotations_file = annotations_file
        self.transform = transform

        if not os.path.isfile(self.annotations_file):
            print(self.annotations_file + 'does not exist!')
        self.file_info = pd.read_csv(annotations_file, index_col=0)
        self.size = len(self.file_info)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image_path = self.file_info['path'][idx]
        if not os.path.isfile(image_path):
            print(image_path + '  does not exist!')
            return None

        image = Image.open(image_path).convert('RGB')
        label_species = int(self.file_info.iloc[idx]['species'])
        label_classes = int(self.file_info.iloc[idx]['classes'])

        sample = {'image': image, 'classes': label_classes,'species': label_species}
        if self.transform:
            sample['image'] = self.transform(image)
        return sample

train_transforms = transforms.Compose([transforms.Resize((500, 500)),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       ])
val_transforms = transforms.Compose([transforms.Resize((500, 500)),
                                     transforms.ToTensor()
                                     ])

train_dataset = MyDataset(root_dir= ROOT_DIR + TRAIN_DIR,
                          annotations_file= TRAIN_ANNO,
                          transform=train_transforms)

test_dataset = MyDataset(root_dir= ROOT_DIR + VAL_DIR,
                         annotations_file= VAL_ANNO,
                         transform=val_transforms)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset)
data_loaders = {'train': train_loader, 'val': test_loader}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def visualize_dataset():
    print(len(train_dataset))
    idx = random.randint(0, len(train_dataset))
    sample = train_loader.dataset[idx]
    print(idx, sample['image'].shape, CLASSES[sample['classes']], SPECIES[sample['species']])
    img = sample['image']
    plt.imshow(transforms.ToPILImage()(img))
    plt.show()
visualize_dataset()

def train_model(model, criterion, optimizer, scheduler, num_epochs=50):
    Loss_list = {'train': [], 'val': []}
    Accuracy_list_results = {'train': [], 'val': []}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-*' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            corrects_results = 0
            corrects_species = 0
            corrects_classes = 0

            for idx,data in enumerate(data_loaders[phase]):
                # print(phase+' processing: {}th batch.'.format(idx))
                inputs = data['image'].to(device)
                labels_species = data['species'].to(device)
                labels_classes = data['classes'].to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    x_output = model(inputs)
                    x_species = x_output[-1].view(-1,3) ### 三类结果
                    x_classes = x_output[-2].view(-1,2) ### 哺乳和禽类结果

                    # _, preds_species = torch.max(x_species, 1)
                    _, preds_species = torch.max(x_species, 1)
                    _, preds_classes = torch.max(x_classes, 1)

                    # criterion = nn.CrossEntropyLoss()
                    # loss = criterion(y_pre, y_train)
                    loss1 = criterion(x_species, labels_species)####计算三分类损失
                    loss2 = criterion(x_classes, labels_classes)###计算损失二分类
                    loss = loss1 + loss2

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

                corrects_species += torch.sum(preds_species == labels_species)
                corrects_classes += torch.sum(preds_classes == labels_classes)
                a = corrects_species
                b = corrects_classes
                c = abs(a-b)
                if a > b:
                    corrects_results = a - c
                elif a < b:
                    corrects_results = b - c
                else:
                    corrects_results = a

                
            epoch_loss = running_loss / (2*len(data_loaders[phase].dataset))
            Loss_list[phase].append(epoch_loss)

            epoch_acc_results = corrects_results.double() / (2*len(data_loaders[phase].dataset))
            epoch_acc = epoch_acc_results

            Accuracy_list_results[phase].append(100 * epoch_acc_results)
            print('{} Loss: {:.4f}  Acc_species: {:.2%}'.format(phase, epoch_loss,epoch_acc_results))

            if phase == 'val' and epoch_acc > best_acc:

                best_acc = epoch_acc_results
                best_model_wts = copy.deepcopy(model.state_dict())
                print('Best val species Acc: {:.2%}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'best_model.pt')
    print('Best val species Acc: {:.2%}'.format(best_acc))
    return model, Loss_list,Accuracy_list_results

network = Net().to(device)
optimizer = optim.Adam(network.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1) # Decay LR by a factor of 0.1 every 1 epochs
model, Loss_list, Accuracy_list_results = train_model(network, criterion, optimizer, exp_lr_scheduler, num_epochs=100)

x = range(100)
y1 = Loss_list["val"]
y2 = Loss_list["train"]

plt.plot(x, y1, color="r", linestyle="-", marker="o", linewidth=1, label="val")
plt.plot(x, y2, color="b", linestyle="-", marker="o", linewidth=1, label="train")
plt.legend()
plt.title('train and val loss vs. epoches')
plt.ylabel('loss')
plt.savefig("train and val loss vs epoches.jpg")
plt.close('all') # 关闭图 0

y5 = Accuracy_list_results["train"]
y6 = Accuracy_list_results["val"]
plt.plot(x, y5, color="r", linestyle="-", marker=".", linewidth=1, label="train")
plt.plot(x, y6, color="b", linestyle="-", marker=".", linewidth=1, label="val")
plt.legend()
plt.title('train and val results acc vs. epoches')
plt.ylabel('Species accuracy')
plt.savefig("train and val results acc vs epoches.jpg")
plt.close('all')

######################################## Visualization ##################################
def visualize_model(model):
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(data_loaders['val']):
            inputs = data['image']
            labels_species = data['species'].to(device)
            labels_classes = data['classes'].to(device)

            x_out = model(inputs.to(device))
            x_species = x_out[1].view( -1,3)
            x_classes = x_out[0].view(-1,2)
            _, preds_species = torch.max(x_species, 1)
            _,preds_classes = torch.max(x_classes,1)

            print(inputs.shape)
            plt.imshow(transforms.ToPILImage()(inputs.squeeze(0)))
            plt.title('predicted species: {}\n ground-truth species:{}'.format(SPECIES[preds_species],SPECIES[labels_species])+'\n'
                      'predicted classes: {}\n ground-truth classes:{}'.format(CLASSES[preds_classes],CLASSES[labels_classes]))
            plt.show()

visualize_model(model)
