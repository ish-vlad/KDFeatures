import os
import tqdm
import numpy as np
import h5py as h5

from shutil import rmtree
from pickle import dump
from open3d import PointCloud, Vector3dVector, KDTreeSearchParamHybrid, \
                    compute_fpfh_feature, estimate_normals, voxel_down_sample

from generate_features_MNIST import config as KD_config
from generate_features_MNIST import generate as generate_KD_features
from lib.KDNet import transform_clouds, to_3D


def make_eval(dataset, path_to_save, rotation_range = None, translation_range = None):
    if rotation_range is not None and translation_range is None:
        #########################
        # Rotation
        #########################
        for alpha in rotation_range:
            # create directory
            directory = create_directory(path_to_save, alpha, 3)
            print(directory)

            # Generate angles from -alpha to alpha
            alphas = alpha * (np.random.rand(len(dataset)) * 2 - 1)

            # from degrees to decimal
            alphas = np.pi / 180 * alphas

            # rotate
            clouds_pair, trans = transform_clouds(dataset, alphas, np.zeros((len(dataset), 1, 2)))

            # save models
            # batch = iteration, transformation,
            #   source_KD, source_FPFH, source_KD_features, source_FPFH_features
            #   target_KD, target_FPFH, target_KD_features, target_FPFH_features (total 10)
            for batch in tqdm.tqdm(get_zip_over_batch(dataset, clouds_pair, trans)):
                i = batch[0]
                name = np.abs(alphas[i] * 180 / np.pi)
                name = 'pair-' + itoa(i, 4) + '_rot-' + itoa(int(name), 3) + '.pkl'

                dump(Store(batch), open(os.path.join(directory, name), 'wb+'))

    elif rotation_range is None and translation_range is not None:
        #########################
        # Translation
        #########################
        for factor in translation_range:
            # create directory
            directory = create_directory(path_to_save, '%.3f' % factor, 0)
            print(directory)

            # Generate shifting params
            rand = np.random.rand(len(dataset), 1, 2)
            factors = factor * (rand * 2 - 1)

            # translate
            clouds_pair, trans = transform_clouds(dataset, np.zeros(len(dataset)), factors)

            # save models
            # batch = iteration, transformation,
            #   source_KD, source_FPFH, source_KD_features, source_FPFH_features
            #   target_KD, target_FPFH, target_KD_features, target_FPFH_features (total 10)
            for batch in tqdm.tqdm(get_zip_over_batch(dataset, clouds_pair, trans)):
                i = batch[0]
                name = np.abs(factors[i])[0]
                name = 'pair-' + itoa(i, 4) + '_trans-' + ('[%.4f,%.4f]' % (name[0], name[1])) + '.pkl'

                dump(Store(batch), open(os.path.join(directory, name), 'wb+'))

    elif rotation_range is not None and translation_range is not None:
        #########################
        # Rotation & Translation
        #########################
        for alpha, factor in zip(rotation_range, translation_range):
            # create directory
            dir_name = itoa(alpha, 3) + '-' + ('%.3f' % factor)
            directory = create_directory(path_to_save, dir_name, 0)
            print(directory)

            # Generate angles from -alpha to alpha
            alphas = alpha * (np.random.rand(len(dataset)) * 2 - 1)

            # from degrees to decimal
            alphas = np.pi / 180 * alphas

            # Generate shifting params
            rand = np.random.rand(len(dataset), 1, 2)
            factors = factor * (rand * 2 - 1)

            # rotate
            clouds_pair, trans = transform_clouds(dataset, alphas, factors)

            # save models
            # batch = iteration, transformation,
            #   source_KD, source_FPFH, source_FPFH_features, source_KD_features_(1|2|3|...)
            #   target_KD, target_FPFH, target_FPFH_features, target_KD_features_(1|2|3|...) (total 8+2*kd_depth)
            for batch in tqdm.tqdm(get_zip_over_batch(dataset, clouds_pair, trans)):
                i = batch[0]
                name = 'pair-' + itoa(i, 4)

                buf = np.abs(alphas[i] * 180 / np.pi)
                name += '_rot-' + itoa(int(buf), 3)

                buf = np.abs(factors[i])[0]
                name += '_trans-' + ('[%.4f,%.4f]' % (buf[0], buf[1]))
                name += '.pkl'

                dump(Store(batch), open(os.path.join(directory, name), 'wb+'))


class Store:
    def __init__(self, batch):
        len_kd = (len(batch) - 8) / 2

        self.pair_num = batch[0]
        self.T_true = batch[1]

        self.source_KD_cloud = batch[2]
        self.source_FPFH_cloud = batch[3]
        self.source_FPFH_features = batch[4]
        self.source_KD_features = batch[5: 5 + len_kd - 1]
        self.source_KD_root = batch[5 + len_kd - 1]

        self.target_KD_cloud = batch[5 + len_kd]
        self.target_FPFH_cloud = batch[6 + len_kd]
        self.target_FPFH_features = batch[7 + len_kd]
        self.target_KD_features = batch[8 + len_kd: 8 + len_kd  + len_kd - 1]
        self.target_KD_root = batch[8 + len_kd + len_kd - 1]


def get_zip_over_batch(source, target, transformation):
    # source_KD, source_FPFH, source_KD_features, source_FPFH_features = source_batch
    source_batch = get_features(source)

    # target_KD, target_FPFH, target_KD_features, target_FPFH_features = target_batch
    target_batch = get_features(target)

    total_batch = (range(len(source)), transformation) + source_batch + target_batch

    return zip(*total_batch)


def to_ply(cloud):
    ply = PointCloud()
    ply.points = Vector3dVector(to_3D(cloud))

    return ply


def get_features(clouds, voxel_size=0.01):
    clouds = to_3D(clouds)
    KDs = generate_KD_features(clouds-clouds.mean(axis=1).reshape(-1,1,clouds.shape[-1]), np.ones(len(clouds)), KD_config)
    clouds_KD = clouds
    clouds_FPFH = []
    FPFHs = []
    for pc in tqdm.tqdm(clouds):
        ply = to_ply(pc)
        ply = voxel_down_sample(ply, voxel_size)
        estimate_normals(ply, KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        fpfh = compute_fpfh_feature(ply, KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
        FPFHs.append(fpfh)
        clouds_FPFH.append(np.asarray(ply.points))

    return (clouds_KD, clouds_FPFH, [f.data for f in FPFHs]) + tuple(KDs)


def itoa(number, digits):
    num_name = str(number)
    while len(num_name) < digits:
        num_name = '0' + num_name
    return num_name


def create_directory(path, number, digits, delete_previous=True):
    num_name = itoa(number, digits)
    directory = os.path.join(path, num_name)

    if os.path.exists(directory):
        if delete_previous:
            rmtree(directory)
            os.mkdir(directory)
        else:
            print(directory, 'already exists!')
    else:
        os.mkdir(directory)
    return directory


if __name__ == "__main__":
    #######################
    # Params initialization
    #######################
    rotation_range = [30, 60, 90, 120, 150, 180]
    translation_range = [0.01, 0.05, 0.1, 0.5, 1.0, 1.5]
    # noise = [0.001, 0.01, 0.1]  # multiply by diameter of the figure TODO check noise param

    path_rotate = '../datasets/MNIST_2D/Rotation/'
    path_translate = '../datasets/MNIST_2D/Translation/'
    path_rotate_and_translate = '../datasets/MNIST_2D/Rotation_and_Translation/'

    with h5.File('../datasets/MNIST_2D/mnist2d.h5', 'r') as hf:
        dataset = np.array(hf.get('X_test'))[:1000, :, :2]

    make_eval(dataset, path_rotate, rotation_range=rotation_range)
    print('Rotation Done')
    make_eval(dataset, path_translate, translation_range=translation_range)
    print('Translation done')
    make_eval(dataset, path_rotate_and_translate, rotation_range=rotation_range, translation_range=translation_range)
    print('Rotation and translation done')
