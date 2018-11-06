import numpy as np
import theano.tensor as T

from lasagne.layers import InputLayer
from lasagne.layers import ReshapeLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import ExpressionLayer
from lasagne.layers import ElemwiseSumLayer, ElemwiseMergeLayer
from lasagne.layers import DenseLayer
from lasagne.layers import BatchNormLayer
from lasagne.nonlinearities import rectify, softmax

import KDNetwork
from KDNetwork.lib.nn.layers import SharedDotLayer, SPTNormReshapeLayer
from KDNetwork.lib.nn.utils import load_weights
from KDNetwork.lib.trees.kdtrees import KDTrees


class KDNetwork:
    def __init__(self, config, model_name='MNIST_2D.v2.1.triplet.pkl'):
        '''
            'RT_AS+TR_2D_MNIST.pkl' - v0
            'MNIST_2D.v1.2.one_direction.pkl' - v1.2
            'MNIST_2D.v1.3.svd.pkl' - v1.3
            'MNIST_2D.v1.4.fpfh.pkl' = v1.4
        '''
        self.clouds = T.tensor3(dtype='float64', name='clouds')
        self.norms = [T.tensor3(dtype='float64', name='norms{}'.format(i)) for i in xrange(config['steps'])]
        self.target = T.vector(dtype='int64')

        self.inputs = [self.clouds, self.norms, self.target]

        KDNet = {}
        if config['input_features'] == 'no':
            KDNet['input'] = InputLayer((None, 1, 2 ** config['steps']), input_var=self.clouds)
        else:
            KDNet['input'] = InputLayer((None, config['dim'], 2 ** config['steps']), input_var=self.clouds)
        for i in xrange(config['steps']):
            KDNet['norm{}_r'.format(i + 1)] = InputLayer((None, 3, 2 ** (config['steps'] - 1 - i)), input_var=self.norms[i])
            KDNet['norm{}_l'.format(i + 1)] = ExpressionLayer(KDNet['norm{}_r'.format(i + 1)], lambda X: -X)

            KDNet['norm{}_l_X-'.format(i + 1)] = SPTNormReshapeLayer(KDNet['norm{}_l'.format(i + 1)], '-', 0,
                                                                     config['n_f'][i + 1])
            KDNet['norm{}_l_Y-'.format(i + 1)] = SPTNormReshapeLayer(KDNet['norm{}_l'.format(i + 1)], '-', 1,
                                                                     config['n_f'][i + 1])
            KDNet['norm{}_l_Z-'.format(i + 1)] = SPTNormReshapeLayer(KDNet['norm{}_l'.format(i + 1)], '-', 2,
                                                                     config['n_f'][i + 1])
            KDNet['norm{}_l_X+'.format(i + 1)] = SPTNormReshapeLayer(KDNet['norm{}_l'.format(i + 1)], '+', 0,
                                                                     config['n_f'][i + 1])
            KDNet['norm{}_l_Y+'.format(i + 1)] = SPTNormReshapeLayer(KDNet['norm{}_l'.format(i + 1)], '+', 1,
                                                                     config['n_f'][i + 1])
            KDNet['norm{}_l_Z+'.format(i + 1)] = SPTNormReshapeLayer(KDNet['norm{}_l'.format(i + 1)], '+', 2,
                                                                     config['n_f'][i + 1])
            KDNet['norm{}_r_X-'.format(i + 1)] = SPTNormReshapeLayer(KDNet['norm{}_r'.format(i + 1)], '-', 0,
                                                                     config['n_f'][i + 1])
            KDNet['norm{}_r_Y-'.format(i + 1)] = SPTNormReshapeLayer(KDNet['norm{}_r'.format(i + 1)], '-', 1,
                                                                     config['n_f'][i + 1])
            KDNet['norm{}_r_Z-'.format(i + 1)] = SPTNormReshapeLayer(KDNet['norm{}_r'.format(i + 1)], '-', 2,
                                                                     config['n_f'][i + 1])
            KDNet['norm{}_r_X+'.format(i + 1)] = SPTNormReshapeLayer(KDNet['norm{}_r'.format(i + 1)], '+', 0,
                                                                     config['n_f'][i + 1])
            KDNet['norm{}_r_Y+'.format(i + 1)] = SPTNormReshapeLayer(KDNet['norm{}_r'.format(i + 1)], '+', 1,
                                                                     config['n_f'][i + 1])
            KDNet['norm{}_r_Z+'.format(i + 1)] = SPTNormReshapeLayer(KDNet['norm{}_r'.format(i + 1)], '+', 2,
                                                                     config['n_f'][i + 1])

            KDNet['cloud{}'.format(i + 1)] = SharedDotLayer(KDNet['input'], config['n_f'][i]) if i == 0 else \
                ElemwiseSumLayer([KDNet['cloud{}_l_X-_masked'.format(i)],
                                  KDNet['cloud{}_l_Y-_masked'.format(i)],
                                  KDNet['cloud{}_l_Z-_masked'.format(i)],
                                  KDNet['cloud{}_l_X+_masked'.format(i)],
                                  KDNet['cloud{}_l_Y+_masked'.format(i)],
                                  KDNet['cloud{}_l_Z+_masked'.format(i)],
                                  KDNet['cloud{}_r_X-_masked'.format(i)],
                                  KDNet['cloud{}_r_Y-_masked'.format(i)],
                                  KDNet['cloud{}_r_Z-_masked'.format(i)],
                                  KDNet['cloud{}_r_X+_masked'.format(i)],
                                  KDNet['cloud{}_r_Y+_masked'.format(i)],
                                  KDNet['cloud{}_r_Z+_masked'.format(i)]])
            KDNet['cloud{}_bn'.format(i + 1)] = BatchNormLayer(KDNet['cloud{}'.format(i + 1)])
            KDNet['cloud{}_relu'.format(i + 1)] = NonlinearityLayer(KDNet['cloud{}_bn'.format(i + 1)], rectify)

            KDNet['cloud{}_r'.format(i + 1)] = ExpressionLayer(KDNet['cloud{}_relu'.format(i + 1)],
                                                               lambda X: X[:, :, 1::2],
                                                               (None, config['n_f'][i], 2 ** (config['steps'] - i - 1)))
            KDNet['cloud{}_l'.format(i + 1)] = ExpressionLayer(KDNet['cloud{}_relu'.format(i + 1)],
                                                               lambda X: X[:, :, ::2],
                                                               (None, config['n_f'][i], 2 ** (config['steps'] - i - 1)))
            #####################
            ### One direction ###
            #####################

            # KDNet['cloud{}_l_X-'.format(i + 1)] = SharedDotLayer(KDNet['cloud{}_l'.format(i + 1)], config['n_f'][i + 1])
            # KDNet['cloud{}_l_Y-'.format(i + 1)] = SharedDotLayer(KDNet['cloud{}_l'.format(i + 1)], config['n_f'][i + 1],
            #                                                      W=KDNet['cloud{}_l_X-'.format(i + 1)].W,
            #                                                      b=KDNet['cloud{}_l_X-'.format(i + 1)].b)
            # KDNet['cloud{}_l_Z-'.format(i + 1)] = SharedDotLayer(KDNet['cloud{}_l'.format(i + 1)], config['n_f'][i + 1],
            #                                                      W=KDNet['cloud{}_l_X-'.format(i + 1)].W,
            #                                                      b=KDNet['cloud{}_l_X-'.format(i + 1)].b)
            # KDNet['cloud{}_l_X+'.format(i + 1)] = SharedDotLayer(KDNet['cloud{}_l'.format(i + 1)], config['n_f'][i + 1])
            # KDNet['cloud{}_l_Y+'.format(i + 1)] = SharedDotLayer(KDNet['cloud{}_l'.format(i + 1)], config['n_f'][i + 1],
            #                                                      W=KDNet['cloud{}_l_X+'.format(i + 1)].W,
            #                                                      b=KDNet['cloud{}_l_X+'.format(i + 1)].b)
            # KDNet['cloud{}_l_Z+'.format(i + 1)] = SharedDotLayer(KDNet['cloud{}_l'.format(i + 1)], config['n_f'][i + 1],
            #                                                      W=KDNet['cloud{}_l_X+'.format(i + 1)].W,
            #                                                      b=KDNet['cloud{}_l_X+'.format(i + 1)].b)
            # KDNet['cloud{}_r_X-'.format(i + 1)] = SharedDotLayer(KDNet['cloud{}_r'.format(i + 1)], config['n_f'][i + 1],
            #                                                      W=KDNet['cloud{}_l_X-'.format(i + 1)].W,
            #                                                      b=KDNet['cloud{}_l_X-'.format(i + 1)].b)
            # KDNet['cloud{}_r_Y-'.format(i + 1)] = SharedDotLayer(KDNet['cloud{}_r'.format(i + 1)], config['n_f'][i + 1],
            #                                                      W=KDNet['cloud{}_l_X-'.format(i + 1)].W,
            #                                                      b=KDNet['cloud{}_l_X-'.format(i + 1)].b)
            # KDNet['cloud{}_r_Z-'.format(i + 1)] = SharedDotLayer(KDNet['cloud{}_r'.format(i + 1)], config['n_f'][i + 1],
            #                                                      W=KDNet['cloud{}_l_X-'.format(i + 1)].W,
            #                                                      b=KDNet['cloud{}_l_X-'.format(i + 1)].b)
            # KDNet['cloud{}_r_X+'.format(i + 1)] = SharedDotLayer(KDNet['cloud{}_r'.format(i + 1)], config['n_f'][i + 1],
            #                                                      W=KDNet['cloud{}_l_X+'.format(i + 1)].W,
            #                                                      b=KDNet['cloud{}_l_X+'.format(i + 1)].b)
            # KDNet['cloud{}_r_Y+'.format(i + 1)] = SharedDotLayer(KDNet['cloud{}_r'.format(i + 1)], config['n_f'][i + 1],
            #                                                      W=KDNet['cloud{}_l_X+'.format(i + 1)].W,
            #                                                      b=KDNet['cloud{}_l_X+'.format(i + 1)].b)
            # KDNet['cloud{}_r_Z+'.format(i + 1)] = SharedDotLayer(KDNet['cloud{}_r'.format(i + 1)], config['n_f'][i + 1],
            #                                                      W=KDNet['cloud{}_l_X+'.format(i + 1)].W,
            #                                                      b=KDNet['cloud{}_l_X+'.format(i + 1)].b)
            #######################
            ### Three direction ###
            #######################

            KDNet['cloud{}_l_X-'.format(i + 1)] = SharedDotLayer(KDNet['cloud{}_l'.format(i + 1)], config['n_f'][i + 1])
            KDNet['cloud{}_l_Y-'.format(i + 1)] = SharedDotLayer(KDNet['cloud{}_l'.format(i + 1)], config['n_f'][i + 1])
            KDNet['cloud{}_l_Z-'.format(i + 1)] = SharedDotLayer(KDNet['cloud{}_l'.format(i + 1)], config['n_f'][i + 1])
            KDNet['cloud{}_l_X+'.format(i + 1)] = SharedDotLayer(KDNet['cloud{}_l'.format(i + 1)], config['n_f'][i + 1])
            KDNet['cloud{}_l_Y+'.format(i + 1)] = SharedDotLayer(KDNet['cloud{}_l'.format(i + 1)], config['n_f'][i + 1])
            KDNet['cloud{}_l_Z+'.format(i + 1)] = SharedDotLayer(KDNet['cloud{}_l'.format(i + 1)], config['n_f'][i + 1])
            KDNet['cloud{}_r_X-'.format(i + 1)] = SharedDotLayer(KDNet['cloud{}_r'.format(i + 1)], config['n_f'][i + 1],
                                                                 W=KDNet['cloud{}_l_X-'.format(i + 1)].W,
                                                                 b=KDNet['cloud{}_l_X-'.format(i + 1)].b)
            KDNet['cloud{}_r_Y-'.format(i + 1)] = SharedDotLayer(KDNet['cloud{}_r'.format(i + 1)], config['n_f'][i + 1],
                                                                 W=KDNet['cloud{}_l_Y-'.format(i + 1)].W,
                                                                 b=KDNet['cloud{}_l_Y-'.format(i + 1)].b)
            KDNet['cloud{}_r_Z-'.format(i + 1)] = SharedDotLayer(KDNet['cloud{}_r'.format(i + 1)], config['n_f'][i + 1],
                                                                 W=KDNet['cloud{}_l_Z-'.format(i + 1)].W,
                                                                 b=KDNet['cloud{}_l_Z-'.format(i + 1)].b)
            KDNet['cloud{}_r_X+'.format(i + 1)] = SharedDotLayer(KDNet['cloud{}_r'.format(i + 1)], config['n_f'][i + 1],
                                                                 W=KDNet['cloud{}_l_X+'.format(i + 1)].W,
                                                                 b=KDNet['cloud{}_l_X+'.format(i + 1)].b)
            KDNet['cloud{}_r_Y+'.format(i + 1)] = SharedDotLayer(KDNet['cloud{}_r'.format(i + 1)], config['n_f'][i + 1],
                                                                 W=KDNet['cloud{}_l_Y+'.format(i + 1)].W,
                                                                 b=KDNet['cloud{}_l_Y+'.format(i + 1)].b)
            KDNet['cloud{}_r_Z+'.format(i + 1)] = SharedDotLayer(KDNet['cloud{}_r'.format(i + 1)], config['n_f'][i + 1],
                                                                 W=KDNet['cloud{}_l_Z+'.format(i + 1)].W,
                                                                 b=KDNet['cloud{}_l_Z+'.format(i + 1)].b)

            KDNet['cloud{}_l_X-_masked'.format(i + 1)] = ElemwiseMergeLayer([KDNet['cloud{}_l_X-'.format(i + 1)],
                                                                             KDNet['norm{}_l_X-'.format(i + 1)]], T.mul)
            KDNet['cloud{}_l_Y-_masked'.format(i + 1)] = ElemwiseMergeLayer([KDNet['cloud{}_l_Y-'.format(i + 1)],
                                                                             KDNet['norm{}_l_Y-'.format(i + 1)]], T.mul)
            KDNet['cloud{}_l_Z-_masked'.format(i + 1)] = ElemwiseMergeLayer([KDNet['cloud{}_l_Z-'.format(i + 1)],
                                                                             KDNet['norm{}_l_Z-'.format(i + 1)]], T.mul)
            KDNet['cloud{}_l_X+_masked'.format(i + 1)] = ElemwiseMergeLayer([KDNet['cloud{}_l_X+'.format(i + 1)],
                                                                             KDNet['norm{}_l_X+'.format(i + 1)]], T.mul)
            KDNet['cloud{}_l_Y+_masked'.format(i + 1)] = ElemwiseMergeLayer([KDNet['cloud{}_l_Y+'.format(i + 1)],
                                                                             KDNet['norm{}_l_Y+'.format(i + 1)]], T.mul)
            KDNet['cloud{}_l_Z+_masked'.format(i + 1)] = ElemwiseMergeLayer([KDNet['cloud{}_l_Z+'.format(i + 1)],
                                                                             KDNet['norm{}_l_Z+'.format(i + 1)]], T.mul)
            KDNet['cloud{}_r_X-_masked'.format(i + 1)] = ElemwiseMergeLayer([KDNet['cloud{}_r_X-'.format(i + 1)],
                                                                             KDNet['norm{}_r_X-'.format(i + 1)]], T.mul)
            KDNet['cloud{}_r_Y-_masked'.format(i + 1)] = ElemwiseMergeLayer([KDNet['cloud{}_r_Y-'.format(i + 1)],
                                                                             KDNet['norm{}_r_Y-'.format(i + 1)]], T.mul)
            KDNet['cloud{}_r_Z-_masked'.format(i + 1)] = ElemwiseMergeLayer([KDNet['cloud{}_r_Z-'.format(i + 1)],
                                                                             KDNet['norm{}_r_Z-'.format(i + 1)]], T.mul)
            KDNet['cloud{}_r_X+_masked'.format(i + 1)] = ElemwiseMergeLayer([KDNet['cloud{}_r_X+'.format(i + 1)],
                                                                             KDNet['norm{}_r_X+'.format(i + 1)]], T.mul)
            KDNet['cloud{}_r_Y+_masked'.format(i + 1)] = ElemwiseMergeLayer([KDNet['cloud{}_r_Y+'.format(i + 1)],
                                                                             KDNet['norm{}_r_Y+'.format(i + 1)]], T.mul)
            KDNet['cloud{}_r_Z+_masked'.format(i + 1)] = ElemwiseMergeLayer([KDNet['cloud{}_r_Z+'.format(i + 1)],
                                                                             KDNet['norm{}_r_Z+'.format(i + 1)]], T.mul)

        KDNet['cloud_fin'] = ElemwiseSumLayer([KDNet['cloud{}_l_X-_masked'.format(config['steps'])],
                                               KDNet['cloud{}_l_Y-_masked'.format(config['steps'])],
                                               KDNet['cloud{}_l_Z-_masked'.format(config['steps'])],
                                               KDNet['cloud{}_l_X+_masked'.format(config['steps'])],
                                               KDNet['cloud{}_l_Y+_masked'.format(config['steps'])],
                                               KDNet['cloud{}_l_Z+_masked'.format(config['steps'])],
                                               KDNet['cloud{}_r_X-_masked'.format(config['steps'])],
                                               KDNet['cloud{}_r_Y-_masked'.format(config['steps'])],
                                               KDNet['cloud{}_r_Z-_masked'.format(config['steps'])],
                                               KDNet['cloud{}_r_X+_masked'.format(config['steps'])],
                                               KDNet['cloud{}_r_Y+_masked'.format(config['steps'])],
                                               KDNet['cloud{}_r_Z+_masked'.format(config['steps'])]])
        KDNet['cloud_fin_bn'] = BatchNormLayer(KDNet['cloud_fin'])
        KDNet['cloud_fin_relu'] = NonlinearityLayer(KDNet['cloud_fin_bn'], rectify)
        KDNet['cloud_fin_reshape'] = ReshapeLayer(KDNet['cloud_fin_relu'], (-1, config['n_f'][-1]))
        KDNet['output'] = DenseLayer(KDNet['cloud_fin_reshape'], config['n_output'], nonlinearity=softmax)

        # Loading weights (optional)
        if model_name is not None:
            load_weights('/home/ishvlad/KDFeatures/workspace/lib/KDNetwork/models/' + model_name, KDNet['cloud_fin_reshape'])

        self.net = KDNet


def iterate_minibatches(*arrays, **kwargs):
    if kwargs['mode'] == 'train':
        indices = np.random.choice(len(arrays[0]), size = len(arrays[0]) - len(arrays[0]) % kwargs['batchsize'])
    elif kwargs['mode'] == 'test':
        indices = np.arange(len(arrays[0]))

    if kwargs['shuffle']:
        np.random.shuffle(indices)

    for start_idx in xrange(0, len(indices), kwargs['batchsize']):
        excerpt = indices[start_idx:start_idx + kwargs['batchsize']]
        tmp = np.transpose(to_3D(arrays[0][excerpt]), axes=(0,2,1))

        if kwargs['augment']:
            # random flipping along x and y axes
            if kwargs['flip']:
                flip = np.random.random(size=(len(tmp), 2, 1))
                flip[flip >= 0.5] = 1.
                flip[flip < 0.5] = -1.
                tmp[:, :2] *= flip

            # stretch picture
            if kwargs['ascale']:
                tmp *= (kwargs['as_min'] + (kwargs['as_max'] - kwargs['as_min'])*np.random.random(size=(len(tmp), 3, 1)))
                tmp = tmp / np.fabs(tmp).max(axis=(1, 2), keepdims=True)

            if kwargs['rotate']:
                r = np.sqrt((tmp[:, :2]**2).sum(axis=1))
                coss = tmp[:, 0]/r
                sins = tmp[:, 1]/r

                if kwargs['test_pos'] is not None:
                    alpha = 2*np.pi*kwargs['test_pos']/kwargs['r_positions']
                else:
                    alpha = 2*np.pi*np.random.randint(0, kwargs['r_positions'], (len(tmp), 1))/kwargs['positions']

                cosr = np.cos(alpha)
                sinr = np.sin(alpha)
                cos = coss*cosr - sins*sinr
                sin = sins*cosr + sinr*coss
                tmp[:, 0] = r*cos
                tmp[:, 1] = r*sin

            if kwargs['mRotate']:
                alphas = np.pi * (np.random.rand(len(tmp)) * 2 - 1)
                tmp, _ = transform_clouds(tmp.transpose(0,2,1)[:,:,:2], alphas, np.zeros((len(tmp), 1, 2)))
                tmp = to_3D(tmp).transpose(0,2,1)

            # translate along all axes
            if kwargs['translate']:
                mins = tmp.min(axis=2, keepdims=True)
                maxs = tmp.max(axis=2, keepdims=True)
                rngs = maxs - mins
                tmp += kwargs['t_rate']*(np.random.random(size=(len(tmp), kwargs['dim'], 1)) - 0.5)*rngs

        tmp_coords = tmp.copy()

        # return true coords for coordinate matchinf
        # batch only for kD tree

        # for centered
        # bias = tmp.mean(axis=-1).reshape(-1, tmp.shape[1], 1)
        # tmp -= bias
        # # for centered and oriented
        # u, s, v = np.linalg.svd(tmp[:,:2].transpose(0, 2, 1))
        # tmp[:, :2] = np.einsum('nik,nij->njk', tmp[:, :2], v)
        # tmp += bias

        trees_data = KDTrees(tmp, dim=3, steps=kwargs['steps'],
                                 lim=kwargs['lim'], det=kwargs['det'], gamma=kwargs['gamma'], medians=True)

        sortings, normals = trees_data['sortings'], trees_data['normals']

        if kwargs['input_features'] == 'all':
            clouds = np.empty((len(excerpt), 3, 2**kwargs['steps']), dtype=np.float32)
            for i, srt in enumerate(sortings):
                clouds[i] = tmp[i, :, srt].T
        elif kwargs['input_features'] == 'no':
            clouds = np.ones((len(excerpt), 1, 2**kwargs['steps']), dtype=np.float32)

        if kwargs['mode'] == 'train':
            yield [clouds] + normals[::-1] + [arrays[1][excerpt]], tmp_coords
        if kwargs['mode'] == 'test':
            yield [clouds] + normals[::-1] + [excerpt], tmp_coords


def get_rotation_matrix(alpha):
    return np.array([
        [np.cos(alpha), np.sin(alpha)],
        [-np.sin(alpha), np.cos(alpha)]
    ])


def transform_clouds(clouds, alpha, vector):
    # center clouds (for rotation)
    bias = np.mean(clouds, axis=1).reshape(-1, 1, clouds.shape[-1])
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


def to_3D(pc_2d):
    if pc_2d.shape[-1] == 3:
        return pc_2d
    return np.concatenate((pc_2d, np.zeros(pc_2d.shape[:-1] + (1,))), axis=-1)