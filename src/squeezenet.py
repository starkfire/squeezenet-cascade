import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import torchvision.transforms as transforms

from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
from PIL import Image
import os
import math
import matplotlib.pyplot as plt
import time


class CustomDataset(Dataset):

    def __init__(self, X, y, BatchSize, transform):
        super().__init__()
        self.BatchSize = BatchSize
        self.X = X
        self.y = y
        self.transform = transform


    def get_number_of_batches(self):
        return math.floor(len(self.list_IDs) / self.BatchSize)
    

    def __getitem__(self, idx):
        class_id = self.y[idx]
        img = Image.open(self.X[idx])
        img = img.convert("RGBA").convert("RGB")
        img = self.transform(img)

        return img, torch.tensor(int(class_id))


    def __len__(self):
        return len(self.X)



class SqueezeNet:

    def __init__(self, path_to_dataset='dataset', class_ids=["bird", "cockatiel"], epochs=10, batch_size=4):
        self.path_to_dataset = path_to_dataset
        self.class_ids = class_ids

        self.labels = pd.DataFrame()
        self.stages = ['train', 'val', 'test']

        # percentage of dataset used for training, validation, and testing set
        self.train_ratio = 0.80
        self.val_ratio = 0.10
        self.test_ratio = 0.10

        # higher batch size = higher memory consumption = faster training
        # lower batch size = less memory consumption = slower training = more detail
        self.batch_size = batch_size

        # number of iterations around the dataset
        self.epochs = epochs
        
        # use SqueezeNet with pre-trained weights from the ImageNet dataset
        self.model = models.squeezenet1_1(pretrained=True)
        
        # use the CPU if this script is executed on a non-CUDA-enabled machine
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def load_custom_model(self, path_to_pt):
        """
        Use a custom SqueezeNet model in form of a .pt file
        """
        self.model = torch.load(path_to_pt, map_location=self.device)


    def export_model(self, path_to_pt):
        """
        Export the current SqueezeNet model as a .pt file
        """
        torch.save(self.model, path_to_pt)


    def generate_labels(self, display=False):
        """
        Create a label map for the dataset
        """
        X = []
        y = []
        filenames = []

        for class_id in self.class_ids:
            for file in os.listdir(os.path.join(os.getcwd(), self.path_to_dataset, class_id)):
                X.append(os.path.join(os.getcwd(), self.path_to_dataset, class_id, file))
                y.append(class_id)
                filenames.append(file)

        self.labels = pd.DataFrame(list(zip(X, filenames, y)), columns=['fileloc', 'filename', 'classid'])
        self.labels['int_class_id'] = self.labels['classid'].astype('category').cat.codes

        if display:
            print(self.labels.head())
            print(self.labels.tail())


    def create_dataset(self):
        """
        Prepare the training, validation, and testing datasets.
        """
        df = self.labels.sample(frac=1)
        X = df.iloc[:, 0]
        y = df.iloc[:, 3]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - self.train_ratio, stratify=y, random_state=0)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=self.test_ratio/(self.test_ratio + self.val_ratio), random_state=0)

        transform = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        image_datasets = {
            'train': CustomDataset(X_train.values, y_train.values, self.batch_size, transform),
            'val': CustomDataset(X_val.values, y_val.values, self.batch_size, transform),
            'test': CustomDataset(X_test.values, y_test.values, self.batch_size, transform)
        }

        self.dataloaders = {
            x: DataLoader(image_datasets[x], batch_size=image_datasets[x].BatchSize, shuffle=True, num_workers=0) for x in self.stages
        }

        self.dataset_sizes = {x: len(image_datasets[x]) for x in self.stages}


    def train(self, overwrite_model=False):
        """
        Method for training/fine-tuning the SqueezeNet model
        """
        self.model.classifier._modules["1"] = torch.nn.Conv2d(512, 5, kernel_size=(1, 1))
        self.model.num_classes = len(self.class_ids)

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.classifier.parameters():
            param.requires_grad = True

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        return self.__train_model(self.model.to(self.device), criterion, optimizer, scheduler, overwrite_model)


    def __train_model(self, model, criterion, optimizer, scheduler, overwrite_model=False):
        """
        Private method which consists of the primary model training logic.
        """
        start_time = time.time()
        num_epochs = self.epochs

        for epoch in range(num_epochs):
            print("Epoch {}/{}".format(epoch, num_epochs))
            print("-" * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0
                num_batches = 0
                outputs = None

                for inputs, labels in self.dataloaders[phase]:
                    if phase == 'train':
                        num_batches += 1
                        percentage_complete = ((num_batches * self.batch_size) / (self.dataset_sizes[phase])) * 100
                        percentage_complete = np.clip(percentage_complete, 0, 100)
                        print("{:0.2f}".format(percentage_complete), "% \\ complete", end="\r")

                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # reset gradient of optimized tensors
                    optimizer.zero_grad()

                    # forward
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs.float(), labels)

                        # backward
                        # optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                            optimizer.step()

                        # statistics
                        running_loss += loss.item() * inputs.size(0)

                        predicted = torch.max(outputs.data, 1)[1]
                        running_corrects += (predicted == labels).sum()

                    if phase == 'train':
                        scheduler.step()

                    epoch_loss = running_loss / self.dataset_sizes[phase]
                    epoch_acc = running_corrects / self.dataset_sizes[phase]

                    print("Loss: {:.4f} Accuracy: {:.4f}".format(epoch_loss, epoch_acc.item()))
        
        time_elapsed = time.time() - start_time
        print("Training finished in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

        if overwrite_model:
            self.model = model

        return model


    def test(self, image_path, model=None, class_ids=["bird", "cockatiel"], display_probabilities=False):
        """
        Test the model against a test image.
        """
        start_time = time.time()
        trained_model = self.model if model is None else model

        input_image = Image.open(image_path)
        preprocess = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            trained_model.to('cuda')

        with torch.no_grad():
            output = trained_model(input_batch)

        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        categories = self.class_ids if model is None else class_ids

        top_prob, top_category_id = torch.topk(probabilities, len(categories))
        time_elapsed = time.time() - start_time

        print("Label: {}, Probability: {}, Inference Time: {:.0f}m {:.6f}s".format(categories[top_category_id[0]], top_prob[0], time_elapsed // 60, time_elapsed % 60))

        if display_probabilities:
            for i in range(top_prob.size(0)):
                print(categories[top_category_id[i]], top_prob[i].item())

        return {"label": categories[top_category_id[0]],
                "probability": top_prob[0],
                "inference_time": "{:.0f}m {:.6f}s".format(time_elapsed // 60, time_elapsed % 60)}
