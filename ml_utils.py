import pandas as pd
import numpy as np
import math
import sklearn.datasets
import ipywidgets as widgets
import torch

##Seaborn for fancy plots. 
#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["figure.figsize"] = (8,8)

def sklearn_to_df(sklearn_dataset):
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df['target'] = pd.Series(sklearn_dataset.target)
    return df

def to_categorical(value_to_convert): 
    """Converts a class vector (integers) to binary class matrix."""
    #y = value_to_convert.as_type('int')    
    y = np.array(value_to_convert, dtype='int').ravel()
    num_classes = np.max(y) + 1
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.int32)
    categorical[np.arange(n), y] = 1
    #print("Categorical shape: ", categorical.shape)
    #print("Sample categorical output: ", categorical[0])
    return categorical

def loss_accuracy_plots(train_losses, train_accuracy, val_losses=None, val_accuracy=None, plot_size=(12, 5)):
    plt.figure(figsize=plot_size)
    plt.subplot(1, 2, 1)
    sns.lineplot(x=range(1, len(train_losses)+1), y=train_losses, label='Train Loss')
    if val_losses is not None:
        sns.lineplot(x=range(1, len(val_losses)+1), y=val_losses, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    sns.lineplot(x=range(1, len(train_accuracy)+1), y=train_accuracy, label='Train Accuracy')
    if val_accuracy is not None:
        sns.lineplot(x=range(1, len(val_accuracy)+1), y=val_accuracy, label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def processEpoch(model, dataloader, optimizer, loss_criterion, device, doTraining=True):
    if doTraining:
        model.to(device)
        model.train()
    else:
        model.to(device)
        model.eval()

    epoch_loss, epoch_accuracy = 0, 0
    for batch, (dataitems, labels) in enumerate(dataloader):
        dataitems, labels = dataitems.to(device), labels.to(device)

        with torch.set_grad_enabled(doTraining):
            outputs = model(dataitems)
            loss = loss_criterion(outputs, labels)
            epoch_loss += loss.item()

            if doTraining:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            _, preds = torch.max(outputs, dim=1)
            epoch_accuracy += (preds == labels).sum().item() / len(labels)

    return epoch_loss / len(dataloader), epoch_accuracy / len(dataloader)

def generic_train_loop(model, training_dataloader, validation_dataloader, optimizer, loss_criterion, epochs, device, printResults=True):
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    for epoch in range(epochs):
        train_loss, train_accuracy = processEpoch(model, training_dataloader, optimizer, loss_criterion, device, doTraining=True)
        val_loss, val_accuracy = processEpoch(model, validation_dataloader, optimizer, loss_criterion, device, doTraining=False)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        if printResults:
            print(f"Epoch: {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    return train_losses, train_accuracies, val_losses, val_accuracies
class edaDF:
    """
    A class used to perform common EDA tasks

    ...

    Attributes
    ----------
    data : dataframe
        a dataframe on which the EDA will be performed
    target : str
        the name of the target column
    cat : list
        a list of the names of the categorical columns
    num : list
        a list of the names of the numerical columns

    Methods
    -------
    setCat(catList)
        sets the cat variable listing the categorical column names to the list provided in the argument catList
        
        Parameters
        ----------
        catlist : list
            The list of column names that are categorical

    setNum(numList)
        sets the cat variable listing the categorical column names to the list provided in the argument catList
        
        Parameters
        ----------
        numlist : list
            The list of column names that are numerical

    countPlots(self, splitTarg=False, show=True)
        generates countplots for the categorical variables in the dataset 

        Parameters
        ----------
        splitTarg : bool
            If true, use the hue function in the countplot to split the data by the target value
        show : bool
            If true, display the graphs when the function is called. Otherwise the figure is returned.
    
    histPlots(self, splitTarg=False, show=True)
        generates countplots for the categorical variables in the dataset 

        Parameters
        ----------
        splitTarg : bool
            If true, use the hue function in the countplot to split the data by the target value
        show : bool
            If true, display the graphs when the function is called. Otherwise the figure is returned. 

    fullEDA()
        Displays the full EDA process. 
    """
    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.cat = []
        self.num = []

    def info(self):
        return self.data.info()

    def giveTarget(self):
        return self.target
        
    def setCat(self, catList):
        self.cat = catList
    
    def setNum(self, numList):
        self.num = numList

    def countPlots(self, splitTarg=False, show=True):
        n = len(self.cat)
        cols = 2
        figure, ax = plt.subplots(math.ceil(n/cols), cols)
        r = 0
        c = 0
        for col in self.cat:
            if splitTarg == False:
                sns.countplot(data=self.data, x=col, ax=ax[r][c])
            if splitTarg == True:
                sns.countplot(data=self.data, x=col, hue=self.target, ax=ax[r][c])
            c += 1
            if c == cols:
                r += 1
                c = 0
        if show == True:
            figure.show()
        return figure

    def histPlots(self, kde=True, splitTarg=False, show=True):
        n = len(self.num)
        cols = 2
        figure, ax = plt.subplots(math.ceil(n/cols), cols)
        r = 0
        c = 0
        for col in self.num:
            #print("r:",r,"c:",c)
            if splitTarg == False:
                sns.histplot(data=self.data, x=col, kde=kde, ax=ax[r][c])
            if splitTarg == True:
                sns.histplot(data=self.data, x=col, hue=self.target, kde=kde, ax=ax[r][c])
            c += 1
            if c == cols:
                r += 1
                c = 0
        if show == True:
            figure.show()
        return figure

    def fullEDA(self):
        out1 = widgets.Output()
        out2 = widgets.Output()
        out3 = widgets.Output()
        out4 = widgets.Output()

        tab = widgets.Tab(children = [out1, out2, out3])
        tab.set_title(0, 'Info')
        tab.set_title(1, 'Categorical')
        tab.set_title(2, 'Numerical')
        display(tab)

        with out1:
            self.info()

        with out2:
            fig2 = self.countPlots(splitTarg=True, show=False)
            plt.show(fig2)
        
        with out3:
            fig3 = self.histPlots(kde=True, show=False)
            plt.show(fig3)

import os
import urllib
class DownloadFromGithub:

    def __init__(self, filename, folder_url=""):
        self.filename = filename
        self.folder_url = folder_url
    
    def get_dataframe(self, **kwargs):
        url = self.folder_url + self.filename
        df = pd.read_csv(url, **kwargs)
        return df
    
    def downloadToPath(self, path):
        url = self.folder_url + self.filename
        file = urllib.request.urlretrieve(url, os.path.join(path, self.filename))
        return file
    
'''# Play with these for different visualization. 
IMAGE_ALPHA = 0.4
SAL_MAP_ALPHA = 0.9
COLOR_MAP = "jet"

indices = [SAMPLE_IMAGE_INDEX, 25, 192, 66, 123, 45, 78, 150, 450, 993]
fig, axes = plt.subplots(2, 5, figsize=(20, 5))
columns = 5
for ax_idx, img_idx in enumerate(indices):
    sal = cnn_model.getSalienceMaps(sal_image=train_data[img_idx][0])
    sal_img = (sal - sal.min()) / (sal.max() - sal.min() + 1e-10)
    sal_img = sal_img.to('cpu').numpy()
    
    s_image = train_data[img_idx][0]
    s_image = (s_image - s_image.min()) / (s_image.max() - s_image.min() + 1e-10)
    
    if ax_idx >= columns:
        axes[1, ax_idx - columns].imshow(sal_img, cmap=COLOR_MAP, alpha=SAL_MAP_ALPHA)
        axes[1, ax_idx - columns].imshow(s_image.permute(1, 2, 0).cpu().numpy(), alpha=IMAGE_ALPHA)
        axes[1, ax_idx - columns].set_title(f'Class: {train_data.classes[train_data[img_idx][1]]} (idx={img_idx})')
        axes[1, ax_idx - columns].axis('off')
    else:
        axes[0, ax_idx].imshow(sal_img, cmap=COLOR_MAP, alpha=SAL_MAP_ALPHA)
        axes[0, ax_idx].imshow(s_image.permute(1, 2, 0).cpu().numpy(), alpha=IMAGE_ALPHA)
        axes[0, ax_idx].set_title(f'Class: {train_data.classes[train_data[img_idx][1]]} (idx={img_idx})')
        axes[0, ax_idx].axis('off')
plt.show()'''