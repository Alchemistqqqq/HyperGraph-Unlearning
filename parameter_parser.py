import argparse
import importlib

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def load_class_from_module(module_name, class_name):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def parameter_parser():
    """
    A method to parse up command line parameters.
    """
    parser = argparse.ArgumentParser()

    ######################### general parameters ################################
    parser.add_argument('--seed', type=int, default=2024, help='Random seed.')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Device to use for training and inference.')
    parser.add_argument('--dataset', type=str, default='CoauthorshipCora', choices=[
        "Cooking200",
        "CoauthorshipCora",
        "CoauthorshipDBLP",
        "CocitationCora",
        "CocitationCiteseer",
        "Recipe100k",
        "Recipe200k",
        "Amazonreviews"
    ], help='Dataset to use.')
    parser.add_argument('--model', type=str, default='HGNNP', choices=["HGNN", "HGNNP","HNHN"], help='Model to use.')

    ########################## training parameters ###########################
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Training set ratio.')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation set ratio.')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Test set ratio.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay.')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs.')
    parser.add_argument('--num_subgraphs', type=int, default=4, help='Number of partitions for the hypergraph.')

    ########################## forgetting parameters ###########################
    parser.add_argument('--forget_ratio', type=float, default=0.2, help='Ratio of nodes to forget.')

    ########################## GIF-specific parameters ###########################
    parser.add_argument('--scale', type=float, default=25, help='Scaling factor for GIF approximation.')
    parser.add_argument('--apply_scale', type=float, default=0.05, help='Scaling factor for applying gradients.')
    parser.add_argument('--iteration', type=int, default=25, help='Number of iterations for influence function approximation.')
    parser.add_argument('--damp', type=float, default=0.01, help='Damping factor for influence function approximation.')
    parser.add_argument('--method', type=str, default='GIF', choices=['GIF', 'IF'], help='Method for influence function approximation.')

    ########################## randomization parameters ###########################
    parser.add_argument('--random_seed', type=int, default=None, help='Random seed for data split and forget.')

    ########################## aggregation method parameters ###########################
    parser.add_argument('--aggregation_method', type=str, default='majority_voting',
                        choices=['majority_voting', 'average_voting', 'weighted_average_voting'],
                        help='Method to aggregate predictions from sub-models.')

    ########################## distance metric parameters ###########################
    parser.add_argument('--distance_metric', type=str, default='l2_norm',
                        choices=['l2_norm', 'manhattan', 'chebyshev', 'cosine'],
                        help='Distance metric for evaluating attack performance.')

    ########################## alpha parameter ###########################
    parser.add_argument('--alpha', type=float, default=0, help='Alpha parameter for controlling the sampling bias.')

    args = parser.parse_args()
    args = vars(args)

    # Dynamically import data sets and model classes
    args['dataset_class'] = load_class_from_module('dhg.data', args['dataset'])
    args['model_class'] = load_class_from_module('dhg.models', args['model'])

    return args
