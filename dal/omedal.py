from copy import copy
import pandas as pd
import numpy as np
from math import ceil, trunc
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import Subset

from torchvision.transforms import (Compose,
                                    Normalize,
                                    ToTensor,
                                    RandomResizedCrop,
                                    )
from utils import (get_dataset_loaders_al,
                   split_epochs_runs,
                   get_loader,
                   preprocess_input_function,
                   makedir,
                   std_mean,
                   save_model_w_condition,
                   normalize_img_rgb,
                   print_metrics_summary)
import runners.train as runner
import model.proto.push as push
from model.proto.settings import coefs
from model.models_utils import warm_only, joint, last_only
import os
import datetime
import wandb as wb


class Omedal:
    def __init__(self, generator) -> None:
        self.labeled_idx = np.array([], dtype=np.int64)
        self.unlabeled_idx = np.array([], dtype=np.int64)
        self.train_indexes = np.array([], dtype=np.int64)
        self.g = generator

    def clear_all(self):
        self.labeled_idx = np.array([], dtype=np.int64)
        self.unlabeled_idx = np.array([], dtype=np.int64)
        self.train_indexes = np.array([], dtype=np.int64)

    def pick_initial_data_points_to_label(self, config):
        '''
        Pick the initial datapoints for L dataset. These datapoints are randomly choosen from the U set
        The number of instances to compose the initial L dataset are set in the config object
        '''
        points_to_label = torch.randperm(
            len(self.train_indexes), dtype=torch.long)[
                :config.initial_l_size+1]
        # Add picked instances in the L set
        self.labeled_idx = np.concatenate((self.labeled_idx, points_to_label))
        # Remove the instances from U set
        self.unlabeled_idx = np.asarray(
            list((set(self.unlabeled_idx) - set(self.labeled_idx))))
        return points_to_label.numpy()

    def pick_data_points_to_label(self,
                                  dataset,
                                  model,
                                  transform,
                                  #last_push_epoch,
                                  config,
                                  dataset_class):
        '''
        Pick datapoints to be inserted in the L dataset,
        updating the list of labeled and unlabeled remaing examples
        '''
        train_trans = transform[0]
        target_trans = transform[1]
        # Select the already labeled instances fraction to compose L set
        previously_labeled_points = self.get_keep_train_fraction(config)
        unlabeled_dataset = Subset(dataset_class(
            dataset, train_trans, target_trans), self.unlabeled_idx)
        # Create a copy of the config file and change the batch_size to use its temporaly for select new instances. Do not alter the original config file
        config_one_inst = copy(config)
        config_one_inst.batch_size = 1
        unlabeled_loader = get_loader(
            unlabeled_dataset, self.g, config_one_inst, suffle=False)
        # Get label instances in accordance with the search query
        points_to_label = self.get_prototypes_and_distances_for_instances(
            unlabeled_loader, model, #last_push_epoch,
            config, print)
        self.labeled_idx = np.concatenate((self.labeled_idx, points_to_label))
        self.unlabeled_idx = np.asarray(
            list((set(self.unlabeled_idx) - set(self.labeled_idx))))
        # Create the L dataset points list with the previously labeled and new labeled instances
        points_L_dataset = np.concatenate(
            (self.labeled_idx[previously_labeled_points], points_to_label))
        return points_L_dataset, previously_labeled_points, points_to_label

    def get_prototypes_and_distances_for_instances(self,
                                                   unlabeled_loader,
                                                   model,
                                                   #last_push_epoch,
                                                   config,
                                                   log):
        '''
        Function to run the search query over the unlabeled dataset and pick
        the instances to be labeled. This function 
        '''

        model.eval()
        

        points_to_label = []
        if config.al_strategy == 'MC_dropout':
            assert int(config.mc_steps) > 0
            print('Running MC Dropout')
            points_to_label = self.mc_dropout_strategy(unlabeled_loader,
                                                       model,
                                                       int(config.mc_steps),
                                                       config
                                                       )
        elif config.al_strategy == 'random':
            print('Random new label points')
            n_samples = len(unlabeled_loader.dataset)
            print(n_samples)
            num_to_label = np.min([config.num_to_label_al, n_samples])
            print(num_to_label)
            points_to_label = np.random.choice(
                self.unlabeled_idx, num_to_label, replace=False)

        print('New Points to Label: ', points_to_label)
        return points_to_label

    def mc_dropout_strategy(self, unlabeled_loader, model, mc_forward_steps, config):
        '''
        Performs the Monte-Carlo Dropout for uncertainty estimation in the U set. Each 
        instance is evaluated for mc_forward_steps with the trained model M. 
        '''

        def enable_dropout(model):
            '''
            Unfreeze the dropout of the model during the inference
            '''
            for m in model.modules():
                if m.__class__.__name__.startswith('Dropout'):
                    m.train()

        n_samples = len(unlabeled_loader.dataset)
        arr_logits1 = []
        arr_logits2 = []
        indexes = []
        # Evaluate instance i from U with M for mc_i steps
        for mc_i in range(mc_forward_steps):
            # Set the model to eval mode and set only dropout layers to train
            model.eval()
            enable_dropout(model)

            for i, (idx, image, label) in enumerate(unlabeled_loader):
                image = image.to(torch.float).cuda()
                with torch.no_grad():
                    logits, min_distance = model(image)
                    logits = torch.nn.Softmax(dim=1)(logits)
                    indexes.append(idx.cpu().numpy()[0])
                    arr_logits1.append(logits.cpu().numpy()[0][0])
                    arr_logits2.append(logits.cpu().numpy()[0][1])

        dict_df = {'idx': indexes, 'logits1': arr_logits1,
                   'logits2': arr_logits2}
        df = pd.DataFrame(dict_df)
        df_means = df.groupby(['idx'])[['logits1', 'logits2']].mean()
        # df_var = df.groupby(['idx'])[['logits1', 'logits2']].var()
        epsilon = 1e-10
        entropy = df_means * np.log(df_means + epsilon)
        entropy = -entropy.sum(axis=1)
        log_df = np.log(df[['logits1', 'logits2']] + epsilon)
        df[['logits1', 'logits2']] = -df[['logits1', 'logits2']] * log_df
        df['sum'] = df[['logits1', 'logits2']].sum(axis=1)
        df_mean = df.groupby(['idx'])['sum'].agg(['mean'])
        df_mean['mutual_info'] = entropy - df_mean['mean']
        # Select the min value between config.num_to_label or number of samples remaining in U set.
        num_to_label = np.min([config.num_to_label_al, n_samples])
        # Sort the most uncertain instances
        sorted_df = df_mean.sort_values(
            'mutual_info', ascending=False)[:num_to_label]

        return sorted_df.index.values

    def update_train_loader(self, train_dataframe, points_to_label, transforms, config, generator, dataset_class):
        '''
        Update the train loader in every DAL iteration. This function is
        adapted to the ProtoPNet model and receives a train_dataframe and
        three transforms. The first and second are functions of the
        train_loader and push_loader. The train_push_trans does not
        perform any data augmentation. The third transformation is
        for the targets.
        '''

        # Different transforms for each loader. The train_trans include data
        # augmentation procedures.
        train_trans, train_push_trans = transforms[0], transforms[1]
        target_trans = transforms[2]

        train_dataset = dataset_class(
            train_dataframe, train_trans, target_trans)
        train_push_dataset = dataset_class(
            train_dataframe, train_push_trans, target_trans)
        labeled_dataset = Subset(train_dataset, points_to_label)
        unlabeled_dataset = Subset(train_dataset, self.unlabeled_idx)
        labeled_push_dataset = Subset(train_push_dataset, points_to_label)

        labeled_loader, unlabeled_loader, labeled_push_loader = get_dataset_loaders_al([labeled_dataset,
                                                                                        unlabeled_dataset, 
                                                                                        labeled_push_dataset],
                                                                                       generator,
                                                                                       config)
        return labeled_loader, unlabeled_loader, labeled_push_loader

    def get_keep_train_fraction(self, config):
        '''
        Return a fraction of the already labeled instances to include
        in the new L set. 
        '''

        keep_n_examples = int(
            config.online_sample_frac * len(self.labeled_idx))
        if keep_n_examples == 0:
            previously_labeled_points = torch.tensor([], dtype=torch.long)
        else:
            previously_labeled_points = torch.randperm(len(self.labeled_idx),
                                                       dtype=torch.long)[:keep_n_examples]
        return previously_labeled_points.numpy()

    def train(self, dataframes, model, optimizer, loss_func, metric, config, **kwargs):
        '''
        The train function with the routine of the DAL framework. The DAL 
        randomly selects n instances to form the L set at the beginning. 
        At each iteration, the function selects new instances and instances 
        into the L-set until no instances remain in the U set.

        Inputs:
            dataframes: the training, validatio and test sets dataframes.
            model: The learning model M
            optimizer: The warmup, joint and last layer optmizers used byt he model
            loss_func: The loss function used by the model
            metric: A set of metrics used to evaluate the model's performance.
            config: Config file
            **kwargs: Other arguments
        '''

        # Setting variables to track the best runs
        best_no_push = 0
        best_push = 0
        best_last = 0
        best_al_iter = [0, 0, 0]
        df_run = pd.DataFrame()
        proj_name = config.proj_name
        # Flag the first DAL iteraction
        start_run = True
        al_iterations = config.al_iterations
        epochs = config.epochs
        reset_weigths = config.reset_al_weights
        push_freq = config.push_freq
        warm_epochs = config.warm_epochs
        # Coreograph the joint / push / last only epochs 
        epochs_split = split_epochs_runs(epochs, push_freq, warm_epochs)
        last_push_epoch = trunc(epochs / push_freq) * push_freq
        if epochs % push_freq != 0:
            last_push_epoch = last_push_epoch + epochs_split[0]

        transformers = kwargs['transforms']
        train_trans = transformers[0]
        target_trans = transformers[1]
        g = kwargs['generator']
        dataset_class = kwargs['dataset_class']
        train_dataframe = dataframes['train']
        val_dataframe = dataframes['val']
        # Get the std and mean previously calculatade of the current dataset
        std, mean = std_mean[config.dataset]['std'], std_mean[config.dataset]['mean']
        
        val_trans = Compose([
            RandomResizedCrop(
                config.img_size, scale=(0.9, 1.0), ratio=(1, 1), antialias=True),
            ToTensor(),
            Normalize(std=std, mean=mean)
        ])
        # Do not apply normalization to the push dataset
        push_transform = Compose([
            RandomResizedCrop(
                config.img_size, scale=(0.9, 1.0), ratio=(1, 1), antialias=True),
            ToTensor(),
        ])

        val_dataset = dataset_class(val_dataframe, val_trans, target_trans)
        val_loader = get_loader(val_dataset, g, config, suffle=False)

        self.train_indexes = train_dataframe.index
        self.unlabeled_idx = train_dataframe.index
        previously_labeled_points = np.asarray([], dtype=np.int64)
        # Pick the initial instances to compose the L set randomly
        points_to_label = self.pick_initial_data_points_to_label(config)
        new_points = points_to_label
        print('initial points', points_to_label)
        time_str = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        # Calculate how many cycles will be necessary to deplete the U set
        max_al_cost = ceil(len(self.unlabeled_idx) /
                           config.num_to_label_al) + 1
        # Set max_al_cost to be the upper bound of the DAL iterations
        if (al_iterations > max_al_cost) or (al_iterations == -1):
            al_iterations = max_al_cost

        # Initiate the DAL cycle
        for al_iter in range(0, al_iterations):
            print(
                f'===============================\n ALTER INTER: {al_iter+1}|{al_iterations}\n===============================')

            # If not the first DAL iteration, select the new instances based on the search strategy
            if not start_run:
                if len(self.unlabeled_idx) > 0:
                    points_to_label, previously_labeled_points, new_points = self.pick_data_points_to_label(train_dataframe, model,
                                                                                                            [train_trans, target_trans],
                                                                                                            #bb_last_push_epoch,
                                                                                                            config,
                                                                                                            dataset_class)
                else:
                    # U set is empty. End the training
                    print(
                        '\n===========================\nU dataset is empty. Execution finished\n===========================\n')
                    break
            if reset_weigths:
                # Currently not reseting the model parameters at all, i.e., only working with the online approach
                pass

            # Updating the loaders to include the new seleced instances. 
            labeled_loader, unlabeled_loader, labeled_push_loader = self.update_train_loader(train_dataframe,
                                                                                             points_to_label,
                                                                                             [train_trans, 
                                                                                              push_transform, 
                                                                                              target_trans],
                                                                                             config,
                                                                                             g,
                                                                                             dataset_class)

            loaders = {'labeled': labeled_loader,
                       'unlabeled': unlabeled_loader,
                       'labeled_push': labeled_push_loader,
                       'val': val_loader}
            # Create directory to each DAL iter to log all the information about the cycle
            model_dir = f'./saved_models/{proj_name}/run_{time_str}_{wb.run.id}/al_iter{al_iter}/'

            # Log the new points included in the L set during this iteration. Uncomment to log 
            #self.inspect_L_and_Points_to_label(loaders, new_points, model_dir)

            # Run the training cycle of the model M
            curr_epoch, best_accus, df_cicle = self.run_cicle(loaders,
                                                              model,
                                                              optimizer,
                                                              loss_func,
                                                              metric,
                                                              config,
                                                              model_dir=model_dir,
                                                              split_epochs_runs=epochs_split,
                                                              warm_up_on=start_run,
                                                              al_iter=al_iter,
                                                              joint_lr_scheduler=kwargs['joint_lr_scheduler'])
            # dataframe with the metrics of the model run
            df_run = pd.concat([df_run, df_cicle], ignore_index=True)
            last_push_epoch = curr_epoch
    
            cicle_no_push = best_accus[0]
            cicle_push = best_accus[1]
            cicle_last = best_accus[2]
            # Check for the best runs for each stage of saved models
            if cicle_no_push > best_no_push:
                best_no_push = cicle_no_push
                best_al_iter[0] = al_iter
            if cicle_push > best_push:
                best_push = cicle_push
                best_al_iter[1] = al_iter
            if cicle_last > best_last:
                best_last = cicle_last
                best_al_iter[2] = al_iter

            start_run = False
        print('========================== BEST RUNS ============================\n')
        print(
            f'|| Metric: {config.metric_eval}                                   \n'
            f'|| Best no push: {best_no_push}         | Al_iter {best_al_iter[0]} \n'
            f'|| Best push:    {best_push}          | Al_iter {best_al_iter[1]} \n'
            f'|| Best last:    {best_last}          | Al_iter {best_al_iter[2]} \n'
            '==================================================================')
        
        df_run.to_csv(f'grid_csv/{wb.run.id}.csv')

    def inspect_L_and_Points_to_label(self, loaders, points_to_label, model_dir):
        '''
        Function to log the new points to tbe included in the L set during the specific
        DAL iteration. This is an log function only, and not alter any aspect of the training
        '''
        makedir(f'{model_dir}L/')
        L_loader = loaders['labeled']
        np.save(f'{model_dir}points_labeled.npy', points_to_label)
        for i, (idx, image, label) in enumerate(L_loader):
            for j, (id, img, lb) in enumerate(zip(idx, image, label)):
                if int(id.cpu()) in points_to_label:
                    img = torch.permute(img, (1, 2, 0))
                    img = normalize_img_rgb(img.cpu().numpy())
                    y = label.cpu()
                    plt.imsave(f'{model_dir}/L/{lb}_{id}.png',
                               img, vmin=0.0, vmax=255.0)

    def run_cicle(self, loaders, model, optimizers, loss_func, metric, config, **kwargs):
        '''
        Run the model M training cycle. For the ProtoPnet model, its executes the 
        joint, push and last only optimization steps, with the specifield epochs for
        each.

        Params:
            loaders: The train, train_push and validation loaders
            optimizers: The optimizers for jointly, push and last only schemes
            loss_func: The loss_function of the model
            Metric: A list of metrics to be tracked
            config: The config variable
            **kwargs: Diverse parameters, as learning_rate scheduler

        '''
        best_no_push = 0
        best_push = 0
        best_last = 0
        df_cicle = pd.DataFrame()
        from model.proto.settings import prototype_activation_function
        al_iter = kwargs['al_iter']
        model_dir = kwargs['model_dir']
        metric_eval = config.metric_eval
        # A baseline metric threshold needed to save a model.pth
        metric_threshold = config.metric_threshold
        makedir(model_dir)
        img_dir = os.path.join(model_dir, 'img')
        makedir(img_dir)
        prototype_img_filename_prefix = 'prototype-img'
        prototype_self_act_filename_prefix = 'prototype-self-act'
        proto_bound_boxes_filename_prefix = 'bb'
        # Log to print during run
        log = print
        warm_epochs = config.warm_epochs
        epochs = config.epochs
        # Output epochs or last only epochs
        output_epochs = config.output_epochs
        #Specifiy the number of image channels to configure the backbone arch
        in_channels = config.in_channels
        train_loader = loaders['labeled']
        labeled_push_loader = loaders['labeled_push']
        val_loader = loaders['val']
        warm_optimizer, joint_optimizer, last_layer_optimizer = optimizers
        joint_lr_scheduler = kwargs['joint_lr_scheduler'] if kwargs and 'joint_lr_scheduler' in kwargs.keys(
        ) else None
        # Flag to only perform warmup during the first DAL iteration
        warm_up_on = kwargs['warm_up_on']
        epochs_split = kwargs['split_epochs_runs']
        current_epoch = 0
        if not warm_up_on:
            joint(model)
        else:
            print('Warmup')
            warm_only(model)
        epochs_split = [epochs - warm_epochs]

        for i in range(warm_epochs):
            current_epoch += 1
            if joint_lr_scheduler and not warm_up_on:
                joint_lr_scheduler.step()
            train_metrics = runner.train(model,
                                            train_loader,
                                            warm_optimizer,
                                            config,
                                            loss_func=loss_func,
                                            metric=metric,
                                            class_specific=True,
                                            use_l1_mask=True,
                                            coefs=coefs,
                                            log=log)

            eval_metrics = runner.test(model,
                                        val_loader,
                                        metric=metric,
                                        class_specific=True,
                                        log=log)

            eval_metrics.update(train_metrics)
            # Compile the metrics to be logged in a csv at the end of the training
            dict_aux = eval_metrics.copy()
            dict_aux.update(
                {'al_iter': [al_iter], 'curr_epoch': [current_epoch]})
            df_cicle = pd.concat(
                [df_cicle, pd.DataFrame(dict_aux)], ignore_index=True)
            
            # Print and log to the W&B
            print_metrics_summary(eval_metrics)
            wb.log(eval_metrics)
            
            if not warm_up_on:
                accu = eval_metrics[metric_eval]
                # Save model with not in warm_up step and above the minimum eval threshold
                if accu > best_no_push:
                    best_no_push = accu

                    save_model_w_condition(model=model,
                                        model_dir=model_dir,
                                        model_name=str(
                                            current_epoch) + 'no_push',
                                        accu=accu,
                                        target_accu=metric_threshold,
                                        log=log)

        print(epochs_split)
        for i in epochs_split:
            for j in range(i):
                current_epoch += 1
                print(f'======== Current Epoch ========: {current_epoch}')
                # joint(model)
                if joint_lr_scheduler:
                    joint_lr_scheduler.step()
                train_metrics = runner.train(model,
                                             train_loader,
                                             joint_optimizer,
                                             metric=metric,
                                             config=config,
                                             log=log,
                                             class_specific=True,
                                             use_l1_mask=True,
                                             coefs=coefs)

                eval_metrics = runner.test(model,
                                           val_loader,
                                           metric=metric,
                                           class_specific=True,
                                           log=log)

                eval_metrics.update(train_metrics)
                dict_aux = eval_metrics.copy()
                dict_aux.update(
                    {'al_iter': [al_iter], 'curr_epoch': [current_epoch]})
                df_cicle = pd.concat(
                    [df_cicle, pd.DataFrame(dict_aux)], ignore_index=True)
                print('======== Joint Metrics Summary ========')
                print_metrics_summary(eval_metrics)
                print('=======================================')
                wb.log(eval_metrics)

                accu = eval_metrics[metric_eval]
                if accu > best_no_push:
                    best_no_push = accu

                    save_model_w_condition(model=model,
                                        model_dir=model_dir,
                                        model_name=str(
                                            current_epoch) + 'no_push',
                                        accu=accu,
                                        target_accu=metric_threshold,
                                        log=log)

            # Performe the push step
            push.push_prototypes(
                labeled_push_loader,
                prototype_network_parallel=model,
                class_specific=True,
                preprocess_input_function=preprocess_input_function,  # normalize if needed
                prototype_layer_stride=1,  # if not None, prototypes will be saved here
                # if not provided, prototypes saved previously will be overwritten
                root_dir_for_saving_prototypes=img_dir,
                epoch_number=current_epoch,
                prototype_img_filename_prefix=prototype_img_filename_prefix,
                prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
                save_prototype_class_identity=True,
                log=print,
                in_channels=in_channels,
            )
            eval_metrics = runner.test(model,
                                       val_loader,
                                       metric=metric,
                                       class_specific=True,
                                       log=log)
            dict_aux = eval_metrics.copy()
            dict_aux.update(
                {'al_iter': [al_iter], 'curr_epoch': [current_epoch]})
            df_cicle = pd.concat(
                [df_cicle, pd.DataFrame(dict_aux)], ignore_index=True)
            wb.log(eval_metrics)

            accu = eval_metrics[metric_eval]
            if accu > best_push:
                best_push = accu

                save_model_w_condition(model=model,
                                       model_dir=model_dir,
                                       model_name=str(current_epoch) + 'push',
                                       accu=accu,
                                       target_accu=metric_threshold,
                                       log=log)

            print('======== Push Metrics Summary ========')
            print_metrics_summary(eval_metrics, is_push=True)
            print('======================================')
            if prototype_activation_function != 'linear':
                last_only(model)
                for j in range(output_epochs):
                    print('last only')
                    train_metrics = runner.train(model,
                                                 train_loader,
                                                 last_layer_optimizer,
                                                 metric=metric,
                                                 config=config,
                                                 class_specific=True,
                                                 coefs=coefs,
                                                 use_l1_mask=True,
                                                 log=log)

                    eval_metrics = runner.test(model,
                                               val_loader,
                                               metric=metric,
                                               class_specific=True,
                                               log=log)
                    
                    eval_metrics.update(train_metrics)
                    dict_aux = eval_metrics.copy()
                    dict_aux.update(
                        {'al_iter': [al_iter], 'curr_epoch': [current_epoch]})
                    df_cicle = pd.concat(
                        [df_cicle, pd.DataFrame(dict_aux)], ignore_index=True)
                    print('======== Last only Metrics Summary ========')
                    print_metrics_summary(eval_metrics)
                    print('============================================')
                    wb.log(eval_metrics)
                    accu = eval_metrics[metric_eval]
                    
                    if accu > best_last:
                        best_last = accu

                        save_model_w_condition(model=model,
                                               model_dir=model_dir,
                                               model_name=str(current_epoch) + f'{j}_push' + '_last',
                                               accu=accu,
                                               target_accu=metric_threshold,
                                               log=log)

        return current_epoch, [best_no_push, best_push, best_last], df_cicle
