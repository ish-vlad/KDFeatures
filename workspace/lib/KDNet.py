import numpy as np
import theano.gpuarray
theano.gpuarray.use('cuda1')
import theano.tensor as T

from lasagne.layers import InputLayer
from lasagne.layers import ReshapeLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import ExpressionLayer
from lasagne.layers import ElemwiseSumLayer, ElemwiseMergeLayer
from lasagne.layers import DenseLayer
from lasagne.layers import BatchNormLayer
from lasagne.nonlinearities import rectify, softmax

from KDNetwork.lib.nn.layers import SharedDotLayer, SPTNormReshapeLayer
from KDNetwork.lib.nn.utils import load_weights
from KDNetwork.lib.trees.kdtrees import KDTrees


class KDNetwork:
    def __init__(self, config):
        self.clouds = T.tensor3(dtype='float64')
        self.norms = [T.tensor3(dtype='float64') for _ in xrange(config['steps'])]
        self.target = T.vector(dtype='int64')

        self.inputs = [self.clouds, self.norms, self.target]

        KDNet = {}
        if config['input_features'] == 'no':
            KDNet['input'] = InputLayer((None, 1, 2 ** config['steps']), input_var=self.clouds)
        else:
            KDNet['input'] = InputLayer((None, 3, 2 ** config['steps']), input_var=self.clouds)
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
        load_weights('lib/KDNetwork/models/RT_AS+TR_2D_MNIST.pkl', KDNet['output'])

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
        tmp = np.transpose(arrays[0][excerpt], axes=(0,2,1))

        # random flipping along x and y axes
        if kwargs['flip']:
            flip = np.random.random(size=(len(tmp), 2, 1))
            flip[flip >= 0.5] = 1.
            flip[flip < 0.5] = -1.
            tmp[:, :2] *= flip

        # stretch picture
        if kwargs['ascale']:
            tmp *= (kwargs['as_min'] + (kwargs['as_max'] - kwargs['as_min'])*np.random.random(size=(len(tmp), kwargs['dim'], 1)))
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

        # translate along all axes
        if kwargs['translate']:
            mins = tmp.min(axis=2, keepdims=True)
            maxs = tmp.max(axis=2, keepdims=True)
            rngs = maxs - mins
            tmp += kwargs['t_rate']*(np.random.random(size=(len(tmp), kwargs['dim'], 1)) - 0.5)*rngs

        trees_data = KDTrees(tmp[:,:2], dim=kwargs['dim']-1, steps=kwargs['steps'],
                                 lim=kwargs['lim'], det=kwargs['det'], gamma=kwargs['gamma'], medians=True)

        sortings, normals = trees_data['sortings'], trees_data['normals']

        if kwargs['input_features'] == 'all':
            clouds = np.empty((len(excerpt), kwargs['dim'], 2**kwargs['steps']), dtype=np.float32)
            for i, srt in enumerate(sortings):
                clouds[i] = tmp[i, :, srt].T
        elif kwargs['input_features'] == 'no':
            clouds = np.ones((len(excerpt), 1, 2**kwargs['steps']), dtype=np.float32)

        if kwargs['mode'] == 'train':
            yield [clouds] + normals[::-1] + [arrays[1][excerpt]]
        if kwargs['mode'] == 'test':
            yield [clouds] + normals[::-1] + [excerpt]