import torch
import numpy as np
import os
from runners.local_analysis import local_analysis
from runners.global_analysis import global_analysis
from model.proto.log import create_logger
from config_parse import parse_args
from utils import (get_dataset_loaders, set_seed)
from dataset.dbutils import get_dataset

'''
Performs the local or global analysis based on the original code of the ProtoPNet
'''

if __name__ == '__main__':

    log, logclose = create_logger('log_file.log')
    config = parse_args()
    infer_mode = config.infer_mode
    load_model = config.load_model
    num_classes = 2
    
    model = torch.load(load_model)
    model.cuda()
    model.eval()  
    load_model_dir = config.load_model_dir 
    epoch_number_str = 10
    save_dir_path = config.save_dir_path
    prototype_shape = model.module.prototype_shape
    prototype_info = np.load(os.path.join(
        load_model_dir, 'epoch-'+str(epoch_number_str), 'bb'+str(epoch_number_str)+'.npy'))

    
    img_size = config.img_size
    if infer_mode == 'local':
        image_path = config.image_path
        image_label = int(config.image_label)

        max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]
        prototype_img_identity = prototype_info[:, -1]

        log('Prototypes are chosen from ' +
            str(len(set(prototype_img_identity))) + ' number of classes.')
        log('Their class identities are: ' + str(prototype_img_identity))

        # confirm prototype connects most strongly to its own class
        prototype_max_connection = torch.argmax(model.module.last_layer.weight, dim=0)
        prototype_max_connection = prototype_max_connection.cpu().numpy()
        if np.sum(prototype_max_connection == prototype_img_identity) == model.module.num_prototypes:
            log('All prototypes connect most strongly to their respective classes.')
        else:
            log('WARNING: Not all prototypes connect most strongly to their respective classes.')
        print(prototype_max_connection)
        k = 2
        local_analysis(image_path,
                       image_label,
                       model,
                       save_dir_path,
                       k,
                       load_model_dir,
                       config,
                       epoch_number_str,
                       prototype_info,
                       max_dist,
                       prototype_img_identity,
                       prototype_max_connection,
                       log)
    elif infer_mode == 'global':
        g = set_seed(config.seed)
        train_dataset, val_dataset, test_dataset, w_sampler, train_push_dataset = get_dataset(config, load_split=True)
        #deve-se usar o train_push para o global_analysis
        _, _, test_loader, train_push_loader = get_dataset_loaders([train_push_dataset,
                                                               val_dataset,
                                                               test_dataset,
                                                               w_sampler,
                                                               train_push_dataset],
                                                              g,
                                                              config)
        root_dir_for_saving = '/'.join(load_model.split('/')[:-1])
        k = 5
        global_analysis([train_push_loader,
                        test_loader],
                       model,
                       k,
                       root_dir_for_saving,
                        epoch_number_str,
                        load_model_dir,
                        prototype_info,
                        config)