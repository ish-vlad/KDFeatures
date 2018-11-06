# input: X_train (N, 256)
#
# for each point cloud generate KD features from different levels
# and finally for the whole point  cloud
#
# output: several X_train_128x(3+32).py --- 256 points map to 128 points with 3 dimension (X,Y,color) and 32 features

import tqdm
import theano
import h5py as h5
import numpy as np

from lasagne.layers import get_output, get_all_params
from open3d import estimate_normals, KDTreeSearchParamHybrid, compute_fpfh_feature, PointCloud, Vector3dVector

from lib.KDNet import KDNetwork, iterate_minibatches

##############
### Config ###
##############

config = {
    # General
    'mode': 'test',
    'batchsize': 2000,
    'shuffle': False,
    # Augmentations
    'augment': False,
    'flip': False,
    'ascale': True, 'as_min': 0.6667, 'as_max': 1.5,
    'rotate': False, 'r_positions': 12, 'test_pos': None,
    'translate': False, 't_rate': 0.1,
    # Point clouds and kd-trees generation
    'steps': 8,  # also control the depth of the network
    'dim': 3,
    'lim': 1,
    'det': False,
    'gamma': 10.,
    # NN options
    'input_features': 'all',  # 'all' for point coordinates, 'no' for feeding 1's as point features
    'n_f': [16,
            32,  32,
            64,  128,
            128, 256,
            512, 128], # representation sizes
    'n_output': 10,
    'l2': 1e-3,
    'lr': 1e-3,
    'n_ens': 16 # number of predicts (take average)
}

##################
### Generating ###
##################
def to_ply(cloud):
    ply = PointCloud()
    ply.points = Vector3dVector(cloud)

    return ply


def append_FPFH(clouds, voxel_size):
    FPFHs = []
    for pc in tqdm.tqdm(clouds.transpose(0, 2, 1)):
        ply = to_ply(pc)
        # ply = voxel_down_sample(ply, voxel_size)
        estimate_normals(ply, KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        fpfh = compute_fpfh_feature(ply, KDTreeSearchParamHybrid(radius=voxel_size * 10, max_nn=100))
        FPFHs.append(fpfh.data.T)

    FPFHs = np.array(FPFHs).transpose(0, 2, 1)
    return FPFHs


def generate(X, y, config):
    # init KD Network
    model = KDNetwork(config)
    clouds, norms = model.clouds, model.norms
    KDNet = model.net

    all_factors = []
    # for each factor
    for factor in range(1, 8):
        features_det = get_output(KDNet['cloud{}'.format(factor+1)], deterministic=True)
        features_fun = theano.function([clouds] + norms, features_det, on_unused_input='ignore')

        # for each batch of pointclouds
        results = []
        for i, (batch, coords) in enumerate(iterate_minibatches(X, y, **config)):
            # fpfh = append_FPFH(coords, 0.01)
            # batch[0] = np.concatenate((batch[0], fpfh), axis=1)

            # divide by sub-pointclouds
            sub_pcs = coords.reshape(coords.shape[0], 3, -1, 2**factor)

            # take center of mass
            sub_pcs = sub_pcs.mean(axis=-1)

            # take features from KD
            features = features_fun(*(batch[:-1]))

            results.append(np.concatenate((sub_pcs, features), axis=1).transpose(0,2,1))

        all_factors.append(np.concatenate(results, axis=0))

    root = []

    features_det = get_output(KDNet['cloud_fin_bn'], deterministic=True)
    features_fun = theano.function([clouds] + norms, features_det, on_unused_input='ignore')

    for i, (batch, coords) in enumerate(iterate_minibatches(X, y, **config)):
        # fpfh = append_FPFH(coords, 0.01)
        # batch[0] = np.concatenate((batch[0], fpfh), axis=1)

        # take features from KD
        features = features_fun(*(batch[:-1]))
        root.append(features)

    root = np.concatenate(root, axis=0).reshape(len(X), -1)

    return all_factors + [root]


def save(X, X_name, all_factors):
    path = '../datasets/MNIST_2D/kd_features/' + X_name + '/'

    np.save(path + 'X_' + X_name + '_256x3.npy', X)
    np.save(path + 'X_' + X_name + '_128x(3+32).npy', all_factors[0])
    np.save(path + 'X_' + X_name + '_064x(3+32).npy', all_factors[1])
    np.save(path + 'X_' + X_name + '_032x(3+64).npy', all_factors[2])
    np.save(path + 'X_' + X_name + '_016x(3+128).npy', all_factors[3])
    np.save(path + 'X_' + X_name + '_008x(3+128).npy', all_factors[4])
    np.save(path + 'X_' + X_name + '_004x(3+256).npy', all_factors[5])
    np.save(path + 'X_' + X_name + '_002x(3+512).npy', all_factors[6])
    np.save(path + 'X_' + X_name + '_001x128(root).npy', all_factors[7])


if __name__ == '__main__':
    # load data
    with h5.File('../datasets/MNIST_2D/mnist2d.h5', 'r') as hf:
        X_train = np.array(hf.get('X_train'))
        y_train = np.array(hf.get('y_train'))
        X_test = np.array(hf.get('X_test'))
        y_test = np.array(hf.get('y_test'))

    # generating
    features = generate(X_train, y_train, config)
    save(X_train, 'train', features)

    features = generate(X_test, y_test, config)
    save(X_train, 'test', features)


