from glob import glob
import torch
from imblearn.under_sampling import RandomUnderSampler
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.transforms import (Compose,
                                    Normalize,
                                    ToTensor,
                                    RandomResizedCrop,
                                    )
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as splitter
from dataset.messidor import Messidor
from utils import std_mean


class MinMaxNorm(object):
    def __call__(self, A):
        A = A.to(torch.float)
        A -= A.min()
        A /= A.max()
        return A


def get_dataset(config, train_compose=None, train_push_compose=None, val_compose=None, test_compose=None, only_dataframe=False, load_split=False):

    train_dataset = None
    train_push_dataset = None
    val_dataset = None
    test_dataset = None

    basedir_data = config.basedata_dir 


        
    # If already splited, just create the dataframes from the csvs.
    if load_split:
        train_dataframe = pd.read_csv('train_messidor.csv')
        val_dataframe = pd.read_csv('val_messidor.csv')
        test_dataframe = pd.read_csv('test_messidor.csv')
        del train_dataframe[train_dataframe.columns[0]]
        del val_dataframe[val_dataframe.columns[0]]
        del test_dataframe[test_dataframe.columns[0]]
        # Get the standard deviadtion and mean of the training set 
        std, mean = get_std_mean(train_dataframe, config)
        # Update the global std_mean values
        std_mean[config.dataset]['std'] = std
        std_mean[config.dataset]['mean'] = mean
        print('messidor  dataset loading')
        # this alternative just returns the dataframes
        if only_dataframe:
            return train_dataframe, val_dataframe, test_dataframe, None, None
    else:
        # Split the messidor dataset in the train, val and test sets from the config values
        csv_files = glob(f'{basedir_data}/*.csv')
        df = pd.DataFrame()
        for f in csv_files:
            df_f = pd.read_csv(f)

            df_f['Image name'] = basedir_data + '/' + f.split('/')[-1][11:17] + '/' + df_f['Image name']
            df = pd.concat([df, df_f])
        df = df.reset_index(drop=True)
        N = df.shape[0]
        train_frac = 1.0 - config.test_split
        val_frac = 1.0 - config.val_split
        train_idxs, test_idxs = splitter(
            np.arange(N), train_size=train_frac,
            stratify=df['Ophthalmologic department'].values)
        train_dataframe_aux = df.iloc[train_idxs]
        train_dataframe_aux = train_dataframe_aux.reset_index(drop=True)
        train_idxs, val_idxs = splitter(
            np.arange(len(train_idxs)), train_size=val_frac,
            stratify=train_dataframe_aux['Ophthalmologic department'].values)
        
        train_dataframe = train_dataframe_aux.iloc[train_idxs]
        train_dataframe = train_dataframe.reset_index(drop=True)
        val_dataframe = train_dataframe_aux.iloc[val_idxs]
        val_dataframe = val_dataframe.reset_index(drop=True)
        test_dataframe = df.iloc[test_idxs]
        test_dataframe = test_dataframe.reset_index(drop=True)

        # If only_dataframe True, save the splits as csvs and return only the dataframes
        if only_dataframe:
            std, mean = get_std_mean(train_dataframe, config)
            std_mean[config.dataset]['std'] = std
            std_mean[config.dataset]['mean'] = mean
            train_dataframe.to_csv('train_messidor.csv')
            val_dataframe.to_csv('val_messidor.csv')
            test_dataframe.to_csv('test_messidor.csv')
            return train_dataframe, val_dataframe, test_dataframe, None, None
    
    std, mean = get_std_mean(train_dataframe, config)

    img_transform=Compose([
            RandomResizedCrop(
                config.img_size, scale=(0.9, 1.0), ratio=(1, 1)),
            ToTensor(),
            Normalize(std=std, mean=mean)
        ])
    push_transform=Compose([
            RandomResizedCrop(
                config.img_size, scale=(0.9, 1.0), ratio=(1, 1)),
            ToTensor(),
        ])
    train_dataset = Messidor(train_dataframe)
    push_dataset = Messidor(train_dataframe, push_transform)
    val_dataset = Messidor(val_dataframe, img_transform)
    test_dataset = Messidor(test_dataframe, img_transform)

    return train_dataset, val_dataset, test_dataset, None, push_dataset    
    

def split_train_val(train_dataset, labels_attr, config, balance=False):
    X_test = None
    train_size = len(train_dataset)
    val_split = config.val_split
    test_split = config.test_split

    y = train_dataset[labels_attr].to_frame()
    y = np.asarray(y)
    idx_df = np.array(range(train_size)).reshape(-1, 1)
    if val_split == 0:
        return idx_df, None
    if balance:
        under_sampler = RandomUnderSampler(random_state=1)
        idx_df, y = under_sampler.fit_resample(idx_df, y)
        X_train, X_val, y_train, y_val = splitter(
            idx_df, y, stratify=y, test_size=val_split)
        print(np.unique(y_train, return_counts=True))
        X_val, X_test, y_val, y_test = splitter(
            X_val, y_val, stratify=y_val, test_size=0.5)
    else:
        X_train, X_val, y_train, y_val = splitter(
            idx_df, y, stratify=y, test_size=val_split)
        print(np.unique(y_train, return_counts=True))
        X_val, X_test, y_val, y_test = splitter(
            X_val, y_val, stratify=y_val, test_size=0.5)

    return np.squeeze(X_train), np.squeeze(X_val), np.squeeze(X_test)


def get_std_mean(train_df, config, indices=None):
    img_size = config.img_size
    
    if config.dataset == 'messidor':
        img_transform=Compose([
                RandomResizedCrop(
                    img_size, scale=(0.9, 1.0), ratio=(1, 1)),
                ToTensor(),
            ])
        if indices is None:
            train_dataset = Messidor(train_df, img_transform)
        
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=10,
    )

    nimages = 0
    mean = 0.0
    var = 0.0
    nimages = len(train_loader.dataset)
    for i_batch, batch_target in enumerate(train_loader):

        batch = batch_target[1]
        batch = batch.view(batch.size(0), batch.size(1), -1)
        mean += torch.nan_to_num(batch).mean(2).sum(0)
        var += torch.nan_to_num(batch).var(2).sum(0)

    mean /= nimages
    var /= nimages
    std = torch.sqrt(var)
    print(f'std: {std}\n Mean: {mean}')
    
    std_mean[config.dataset] = {'std': torch.Tensor(
        std).numpy(), 'mean': torch.Tensor(mean).numpy()}
    return std, mean


def get_weighted_sampler(df_train, indexes=None, config=None, method='torch'):
    if method == 'torch':
        if indexes is not None:
            df_train = df_train.iloc[indexes]
            counts = np.unique(df_train['label'].values, return_counts=True)[1]
        else:
            counts = np.unique(df_train['label'].values, return_counts=True)[1]
        class_weight = 1./torch.tensor(counts, dtype=torch.float)
        samples_weight = np.array([class_weight[t] for t in df_train['label']])
        samples_weight = torch.from_numpy(samples_weight)
        print(counts)

        weighted_sampler = WeightedRandomSampler(weights=samples_weight.double(),
                                                 num_samples=len(
                                                     samples_weight),
                                                 replacement=True)
        return weighted_sampler
    elif method == 'undersampling':
        r_under = RandomUnderSampler(random_state=config.seed)
        df_train = df_train.iloc[indexes]
        df_train_y = df_train['label'].values
        df_train.drop(columns=['label'], inplace=True)
        # df_train.reset_index(inplace=True)
        df_train_resampled, y_resampled = r_under.fit_resample(
            df_train, df_train_y)
        df_train = df_train_resampled.join(pd.DataFrame(y_resampled,
                                                        columns=['label'],
                                                        index=df_train_resampled.index))

        print(df_train)
        return df_train
