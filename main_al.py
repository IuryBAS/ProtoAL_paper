import torch 
import wandb as wb
from config_parse import parse_args
from dataset.dbutils import get_dataset
from utils import (
    set_seed,
)
from utils import makedir
from dal import omedal as al
from dataset import Messidor
from model.proto.ppnet import construct_PPNet
from model.models_utils import configure_optimizers
from torcheval.metrics import (BinaryAccuracy,
                               BinaryAUPRC,
                               BinaryPrecision,
                               BinaryRecall,                               
                               BinaryF1Score,
                               )

if __name__ == '__main__':
    config = parse_args()
    print(config)
    proj_name = config.proj_name
    num_classes = config.num_classes
    in_channels = config.in_channels
    epochs = config.epochs
    push_freq = config.push_freq
    warm_epochs = config.warm_epochs
    img_size = config.img_size
    output_epochs = config.output_epochs
    arch_model = config.arch_model
    seed = config.seed
    run_op = config.run
    class_specific = config.class_specific
    g = set_seed(seed)
    makedir('grid_csv')
    train_trans, target_trans = None, None

    #Loading dataframes with pre-defined splits
    # To generate the csv files, set load_split to false. The results of
    # the paper were generated with seed 1
    train_dataframe, val_dataframe, test_dataframe, _, _ = get_dataset(
        config, only_dataframe=True, load_split=False)
    dataframes = {'train': train_dataframe,
                  'val': val_dataframe,
                  'test': test_dataframe}
    
    # Building the ProtoPNet model
    ppnet = construct_PPNet(arch_model,
                                img_size=img_size,
                                pretrained=True,
                                num_classes=num_classes,
                                in_channels=in_channels,
                                prototype_shape=(num_classes * 6, 256, 1, 1)) # define 6 prototypes per class
    ppnet = ppnet.cuda()
    # Setting the optimizers for each training phase
    warm_optimizer = configure_optimizers(ppnet, 'warm')
    joint_optimizer, joint_lr_scheduler = configure_optimizers(
        ppnet, 'joint')
    last_layer_optimizer = configure_optimizers(ppnet, 'last')
    
    ppnet_dp = torch.nn.DataParallel(ppnet, device_ids=[config.device])

    loss_fn = torch.nn.CrossEntropyLoss()
    metric = [BinaryAccuracy(),
              BinaryAUPRC(),
              BinaryPrecision(),
              BinaryRecall(),                               
              BinaryF1Score()]
    acl = al.Omedal(g)

    # Initiating the W&B tracker
    wb.init(project=proj_name,
                config=vars(config))
    acl.train(dataframes,
                ppnet_dp,
                [warm_optimizer,
                joint_optimizer,
                last_layer_optimizer],
                loss_fn,
                metric,
                config,
                joint_lr_scheduler=joint_lr_scheduler,
                transforms=[train_trans, target_trans],
                generator=g,
                dataset_class=Messidor
            )
    wb.finish()