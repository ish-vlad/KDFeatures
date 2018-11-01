# make evaluation set
# 1) rotation: uniformly on [-a;a], a = 30, 60, 90, 120, 150, 180
# 2) Scale: uniformly on [-b,b], b = cd, d - figure diameter on corresponding axis, c = 0.01, 0.05, 0.1, 0.5, 1.0, 5.0
# 3) Rotation and scale: 6 experiments, one2one
# 4) Noise: add different noise to data: 0.01, 0.1, 1.0. Play with rotation and scale and noise.

import os
import numpy as np
import h5py as h5

from shutil import rmtree
from pickle import dump

#######################
# Params initialization
#######################

rotation_range = [30, 60, 90, 120, 150, 180]
translation_range = [0.01, 0.05, 0.1, 0.5, 1.0, 1.5]  # multiply by diameter of the figure
noise = [0.001, 0.01, 0.1] # multiply by diameter of the figure TODO check noise param

path_rotate = '../datasets/MNIST_2D/Rotation/'
path_translate = '../datasets/MNIST_2D/Translation/'
path_rotate_and_translate = '../datasets/MNIST_2D/Rotation_and_Translation/'

with h5.File('../datasets/MNIST_2D/mnist2d.h5', 'r') as hf:
    dataset = np.array(hf.get('X_test'))[:1000, :, :2]


def get_rotation_matrix(alpha):
    return np.array([
        [np.cos(alpha), np.sin(alpha)],
        [-np.sin(alpha), np.cos(alpha)]
    ])


def transform_clouds(clouds, alpha, vector):
    # center clouds (for rotation)
    bias = np.mean(clouds, axis=1).reshape(-1, 1, 2)
    clouds = clouds - bias

    # rotation
    Ts = get_rotation_matrix(alpha)
    rotated_clouds = np.einsum('npi,jin->npj', clouds, Ts)
    rotated_clouds += bias

    # translation
    final_clouds = rotated_clouds + vector

    # rotation of bias vector
    bias = bias.reshape(-1, 2)
    bias_rotated = np.einsum('ni,jin->nj', bias, Ts)

    transformation = np.zeros((len(clouds), 4, 4))
    transformation[:, :2, :2] = Ts.transpose(2, 0, 1)
    transformation[:, :2, -1] = vector[:, 0, :] + bias - bias_rotated
    transformation[:, 2, 2] = np.ones(len(clouds))
    transformation[:, -1, -1] = np.ones(len(clouds))

    return final_clouds, transformation


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

#########################
# Rotation
#########################


for alpha in rotation_range:
    # create directory
    directory = create_directory(path_rotate, alpha, 3)

    # Generate angles from -alpha to alpha
    alphas = alpha*(np.random.rand(len(dataset))*2-1)

    # from degrees to decimal
    alphas = np.pi/180 * alphas

    # rotate
    clouds_pair, trans = transform_clouds(dataset, alphas, np.zeros((len(dataset), 1, 2)))

    # save models
    for i, pc, pc_rot, t in zip(range(len(dataset)), dataset, clouds_pair, trans):
        name = np.abs(alphas[i] * 180 / np.pi)
        name = 'pair-' + itoa(i, 4) + '_rot-' + itoa(int(name), 3) + '.pkl'
        dump((pc, pc_rot, t), open(os.path.join(directory, name), 'wb+'))

#########################
# Translation
#########################

for factor in translation_range:
    # create directory
    directory = create_directory(path_translate, '%.3f' % factor, 0)

    # Generate shifting params
    rand = np.random.rand(len(dataset), 1, 2)
    factors = factor*(rand*2-1)

    # translate
    clouds_pair, trans = transform_clouds(dataset, np.zeros(len(dataset)), factors)

    # save models
    for i, pc, pc_rot, t in zip(range(len(dataset)), dataset, clouds_pair, trans):
        name = np.abs(factors[i])[0]
        name = 'pair-' + itoa(i, 4) + '_trans-' + ('[%.4f,%.4f]' % (name[0], name[1])) + '.pkl'
        dump((pc, pc_rot, t), open(os.path.join(directory, name), 'wb+'))

#########################
# Rotation & Translation
#########################

for alpha, factor in zip(rotation_range, translation_range):
    # create directory
    dir_name = itoa(alpha, 3) + '-' + ('%.3f' % factor)
    directory = create_directory(path_rotate_and_translate, dir_name, 0)

    # Generate angles from -alpha to alpha
    alphas = alpha*(np.random.rand(len(dataset))*2-1)

    # from degrees to decimal
    alphas = np.pi/180 * alphas

    # Generate shifting params
    rand = np.random.rand(len(dataset), 1, 2)
    factors = factor*(rand*2-1)

    # rotate
    clouds_pair, trans = transform_clouds(dataset, alphas, factors)

    # save models
    for i, pc, pc_rot, t in zip(range(len(dataset)), dataset, clouds_pair, trans):
        name = 'pair-' + itoa(i, 4)

        buf = np.abs(alphas[i] * 180 / np.pi)
        name += '_rot-' + itoa(int(buf), 3)

        buf = np.abs(factors[i])[0]
        name += '_trans-' + ('[%.4f,%.4f]' % (buf[0], buf[1]))

        name += '.pkl'
        dump((pc, pc_rot, t), open(os.path.join(directory, name), 'wb+'))
