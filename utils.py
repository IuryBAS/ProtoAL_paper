import numpy as np
import os
from colorama import Fore
import torch
from torch.utils.data import DataLoader
import random
from config_parse import parse_args

dir_path = os.path.dirname(os.path.realpath(__file__))
# Pre-calculated std and mean of some datasets
std_mean = {'isic': {'std': [0.1002, 0.1255, 0.1400],
                     'mean': [0.7636, 0.5462, 0.5706]},
            'cub': {'std': [0.229, 0.224, 0.225],
                    'mean': [0.485, 0.456, 0.406]},
            'mnist': {'std': [0.3081],
                      'mean': [0.1307]},
            'messidor': {'std': [0.3189, 0.1570, 0.0601],
                         'mean': [0.4589, 0.2177, 0.0723]},
            }

def makedir(path_dir):
    try:
        os.makedirs(path_dir)
        print(f'OK: Directory {path_dir} created')
    except OSError as error:
        print(Fore.YELLOW + f'Diretório {path_dir} já existe')
        print(error)
        print(Fore.WHITE)

def print_metrics_summary(metrics, mode='all', is_push=False):
    from rich.console import Console
    from rich.table import Table
    metrics_keys = []
    for me in metrics.keys():
        me = me.replace('_train', '')
        me = me.replace('_test', '')
        metrics_keys.append(me)

    table = Table(title="Metrics summary of the run")

    table.add_column("Metric", justify="right", style="cyan", no_wrap=True)
    table.add_column("Train", justify="right", style="magenta")
    table.add_column("Val/Test", justify="right", style="green")
    for m in metrics_keys:
        if is_push:
            table.add_row(m, '-', str(metrics[f'{m}_test']))
        else:
            table.add_row(m, str(metrics[f'{m}_train']), str(metrics[f'{m}_test']))
    
    console = Console()
    console.print(table)


def split_epochs_runs(total_epochs, push_freq, warm_epochs):
    '''
    Function to split the epochs in the context of the ProtoPNet
    model, with its warmup, joint, push and last only train stages.

    Params:
        total_epochs: The total number of epochs to run
        push_freq: Split the total epochs to perform a push/projection
                    stage at the respective frequence
        warm_epochs: The epochs to perform the warmup of the model 
    '''
    assert total_epochs > push_freq
    arr = []
    warm_epochs_aux = warm_epochs
    while sum(arr) < total_epochs:
        warm_epochs_aux -= push_freq
        if warm_epochs_aux <= 0:
            if (sum(arr) + push_freq) > total_epochs:
                arr.append(total_epochs - sum(arr))
                break
            arr.append(push_freq)
    for i, val in enumerate(arr):
        diff = val - warm_epochs
        print(diff)
        if diff <= 0:
            arr.pop(i)
            warm_epochs = abs(diff)
        else:
            arr[0] = diff
            break

    return arr


def find_high_activation_crop(activation_map, percentile=95):
    '''
    Function borrowed from the ProtoPnet code
    '''
    threshold = np.percentile(activation_map, percentile)
    mask = np.ones(activation_map.shape)
    mask[activation_map < threshold] = 0
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > 0.5:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > 0.5:
            upper_y = i
            break
    for j in range(mask.shape[1]):
        if np.amax(mask[:, j]) > 0.5:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:, j]) > 0.5:
            upper_x = j
            break
    return lower_y, upper_y+1, lower_x, upper_x+1


def normalize_img(image, imax=8, dtype=np.uint8):
    '''
        Normalize an image between its maximum and minimum values, and with the
        specifield caracteristics

        Params:
            image: An image to be normalized
            imax: The value of bits to represent the pixel values
            dtype: The desired dtype of the image

        Returns:
            A normalized image
    '''
    img_max = np.max(image)
    img_min = np.min(image)

    # Prevents division by 0 when the img_max and img_min have the same value
    if img_max == img_min:
        img_sub_norm = (image-img_min) / ((img_max - img_min) + 1e-12)

    else:
        img_sub_norm = (image-img_min) / (img_max - img_min)
    # Normalize image accordinly with the maximum bits representation
    # passed as parameter
    img_sub_norm = (img_sub_norm * ((2**imax) - 1)).astype(dtype)
    return img_sub_norm


def normalize_img_rgb(image):
    '''
    Function to perform normalization in RGB images

    Params:
        image: An input image to be normalized
    Returns:
        A normalized image
    '''

    norm_image = np.empty_like(image)

    norm_image[:, :, 0] = normalize_img(image[:, :, 0], dtype='float32')
    norm_image[:, :, 1] = normalize_img(image[:, :, 1], dtype='float32')
    norm_image[:, :, 2] = normalize_img(image[:, :, 2], dtype='float32')

    return norm_image.astype('uint8')


def set_seed(seed, return_dataloader_gen=True):
    '''
    Set the seed for np.random, torch seed, and ramdom seed. Return a
    torch.generator for use in the torch dataloaders
    '''
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    g = torch.Generator()
    g.manual_seed(seed)

    return g



def get_loader(dataset, generator, config, w_sampler=None, **kwargs):
    '''
    Creates a loader with the specifield config and generator. 
    '''
    def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

    assert dataset is not None
    batch_size = config.batch_size
    workers = config.workers
    suffle = kwargs['suffle'] if kwargs['suffle'] else True
    if w_sampler is not None:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=suffle,
            num_workers=workers,
            generator=generator,
            worker_init_fn=seed_worker,
            pin_memory=False  # torch.Generator(device='cpu'),
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=suffle,
            num_workers=workers,
            generator=generator,  # torch.Generator(device='cpu'),
            sampler=w_sampler,
            pin_memory=False,
            worker_init_fn=seed_worker  # torch.Generator(device='cpu'),
        )
    print('Loader set size: {0}'.format(len(loader.dataset)))
    return loader

def get_dataset_loaders(datasets, generator, config):
    '''
    Create dataset loaders in batch, in the context of the ProtoPNet model,
    with train, validation, test and train_push data.
    
    Params
        datasets: Contains a list of datasets to create the loaders. Also,
                the list can include the w_sampler, for dataset balancing.
    '''
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_dataset, val_dataset, test_dataset, w_sampler, train_push = datasets
    assert train_dataset is not None
    batch_size = config.batch_size
    workers = config.workers

    if w_sampler is None:
        print('Loader carregado com W_sampler')
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            generator=generator,
            worker_init_fn=seed_worker,
            pin_memory=True  # torch.Generator(device='cpu'),
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            generator=generator,  # torch.Generator(device='cpu'),
            sampler=w_sampler,
            pin_memory=True,
            worker_init_fn=seed_worker  # torch.Generator(device='cpu'),
        )

    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            # sampler=w_sampler,
            pin_memory=True,
            generator=generator,
            worker_init_fn=seed_worker  # torch.Generator(device='cpu'),

        )
    else:
        val_loader = None
    labels = []
    
    # Uncomment to print check the labels distribution when using 
    # w_sampler
    '''
    for (idx, image, label) in train_loader:
        labels.extend(label)
    print(np.unique(labels, return_counts=True))
    '''

    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=100,  # batch_size,
            shuffle=False,
            num_workers=workers,
            # sampler=w_sampler,
            pin_memory=True,
            generator=generator,
            worker_init_fn=seed_worker  # torch.Generator(device='cpu'),

        )
    else:
        test_loader = None

    # If not specified an train_push dataset, the train dataset will be used to
    # construct the train_push loader. (Note: its only should be performed if the
    # train dataset was not previously augmented or similar procedure)  
    if train_push is not None:
        print('Existe train_push especifico')
        train_push_loader = DataLoader(
            train_push,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            generator=generator,
            worker_init_fn=seed_worker  # torch.Generator(device='cpu'),

        )
    else:
        train_push_loader = DataLoader(
            train_dataset,
            batch_size=75,  # batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            generator=generator,
            worker_init_fn=seed_worker  # torch.Generator(device='cpu'),
        )

    print('training set size: {0}'.format(len(train_loader.dataset)))
    print('push set size: {0}'.format(len(train_push_loader.dataset)))
    print('test set size: {0}'.format(len(test_loader.dataset)))
    print('batch size: {0}'.format(batch_size))

    return train_loader, val_loader, test_loader, train_push_loader

# TODO
def get_dataset_loaders_al(datasets, generator, config):
    '''
    Creates the labeled, unlabeled and labeled_push loaders to the DAL cycle and 
    ProtoPNet
    '''
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    labeled_dataset, unlabeled_dataset, labeled_push_dataset = datasets
    assert labeled_dataset is not None
    assert unlabeled_dataset is not None or len(unlabeled_dataset) > 0
    batch_size = config.batch_size
    workers = config.workers

    labeled_loader = DataLoader(
        labeled_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        generator=generator,  # torch.Generator(device='cpu'),
        pin_memory=True,
        worker_init_fn=seed_worker  # torch.Generator(device='cpu'),
    )
    
    # Uncomment to check how balanced is the L set the DAL step
    '''
    labels = []
    for _, image, label in labeled_loader:
        labels.extend(label)
    print(np.unique(labels, return_counts=True))
    '''

    labeled_push_loader = DataLoader(
        labeled_push_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        generator=generator,  # torch.Generator(device='cpu'),
        pin_memory=True,
        worker_init_fn=seed_worker  # torch.Generator(device='cpu'),
    )
    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        # sampler=w_sampler,
        pin_memory=True,
        generator=generator,
        worker_init_fn=seed_worker  # torch.Generator(device='cpu'),

    )
    print('Labeled set size: {0}'.format(len(labeled_loader.dataset)))
    print('Unlabeled set size: {0}'.format(len(unlabeled_loader.dataset)))

    return labeled_loader, unlabeled_loader, labeled_push_loader


# Function bellow were borrowed from the ProtoPNet code
def preprocess(x, mean, std):
    channels = x.size(1)
    y = torch.zeros_like(x)
    for i in range(channels):
        y[:, i, :, :] = (x[:, i, :, :] - mean[i]) / std[i]
    return y


def preprocess_input_function(x):
    '''
    allocate new tensor like x and apply the normalization used in the
    pretrained model
    '''
    config = parse_args()
    std = std_mean[config.dataset]['std']
    mean = std_mean[config.dataset]['mean']
    
    return preprocess(x, mean=mean, std=std)


def undo_preprocess(x, mean, std):
    channels = x.size(1)
    y = torch.zeros_like(x)
    for i in range(channels):
        y[:, i, :, :] = x[:, i, :, :] * std[i] + mean[i]
    return y


def undo_preprocess_input_function(x):
    '''
    allocate new tensor like x and undo the normalization used in the
    pretrained model
    '''
    config = parse_args()
    std = std_mean[config.dataset]['std']
    mean = std_mean[config.dataset]['mean']

    return undo_preprocess(x, mean=mean, std=std)


def save_model_w_condition(model, model_dir, model_name, accu, target_accu, log=print):
    '''
    model: this is not the multigpu model
    '''
    if accu > target_accu:
        log('\tabove {0:.2f}%'.format(target_accu * 100))
        # torch.save(obj=model.state_dict(), f=os.path.join(model_dir, (model_name + '{0:.4f}.pth').format(accu)))
        torch.save(obj=model, f=os.path.join(
            model_dir, (model_name + '{0:.4f}.pth').format(accu)))
