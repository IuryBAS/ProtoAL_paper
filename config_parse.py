import configargparse as conf


def parse_args():

    parser = conf.ArgParser(default_config_files=['config.ini'])

    parser.add('-c', '--config_file', required=False,
               is_config_file=True, help='Specify a desired config file to use')
    parser.add('-bd', '--basedata_dir', type=str, default=None,
               help='Set the base directory for the dataset')
    parser.add('-d', '--dataset', required=True, default='isic', )
    parser.add('--dataset_meta', default=None)
    parser.add('-is', '--img_size', type=int, default=None)
    parser.add('-m', '--arch_model', type=str, default='resnet18')
    parser.add('--proj_name', type=str, default='Proto')
    parser.add('-run', '--run', type=str, default='train')
    parser.add('-l', '--load_model', type=str, default=None)
    parser.add('-lmd', '--load_model_dir', type=str, default=None)
    parser.add('--device', type=str, default='cuda')
    parser.add('--gpu', type=bool, default=True)
    parser.add('--metric_threshold', type=float, default=.70)
    parser.add('--metric_eval', type=str, default='BinaryF1Score_test')

    parser.add('-e', '--epochs', type=int, default=None)
    parser.add('-b', '--batch_size', type=int, default=None)
    parser.add('-w', '--workers', type=int, default=None)
    parser.add('-s', '--seed', type=int, default=1)

    parser.add('--val_split', type=float, default=0)
    parser.add('--test_split', type=float, default=0)
    parser.add('--validate_interval', type=int, default=1)
    parser.add('-n', '--num_classes', type=int, default=None)
    parser.add('-ic', '--in_channels', type=int, default=None)
    parser.add('-pf', '--push_freq', type=int,  default=10)
    parser.add('-warm', '--warm_epochs', type=int, default=5)
    parser.add('-oe', '--output_epochs', type=int, default=20)
    parser.add('-suffle', '--suffle', type=bool, default=True)
    parser.add('--class_specific', type=bool, default=True)

    parser.add('--infer_mode', type=str, default='local')
    parser.add('--image_path', type=str, default=None)
    parser.add('--save_dir_path', type=str, default=None)
    parser.add('--image_label', type=str, default=None)

    parser.add('--initial_l_size', type=int, default=100)
    parser.add('--num_to_label_al', '--num_points_to_label_per_iter', type=int, default=5)
    parser.add('--al_iterations', type=int, default=5)
    parser.add('--reset_al_weights', type=bool, default=False)
    parser.add('--online_sample_frac', type=float, default=0.87)
    parser.add('--al_strategy', type=str, default='prototype_dist')
    parser.add('--mc_steps', type=int, default=10)

    args = parser.parse_args()
    return args
