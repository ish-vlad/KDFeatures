import numpy as np
import h5py as h5

from lib.KDNetwork.lib.generators.meshgrid import generate_clouds

path2data = '../datasets/ModelNet10/'
depth = 8

with h5.File(path2data + 'modelnet10.h5', 'r') as hf:
    train_vertices = np.array(hf.get('train_vertices'))
    train_faces = np.array(hf.get('train_faces'))
    train_nFaces = np.array(hf.get('train_nFaces'))
    train_labels = np.array(hf.get('train_labels'))
    test_vertices = np.array(hf.get('test_vertices'))
    test_faces = np.array(hf.get('test_faces'))
    test_nFaces = np.array(hf.get('test_nFaces'))
    test_labels = np.array(hf.get('test_labels'))

X_train = generate_clouds(np.arange(len(train_labels)), depth, train_vertices, train_faces, train_nFaces)
X_test = generate_clouds(np.arange(len(test_labels)), depth, test_vertices, test_faces, test_nFaces)
y_train = train_labels
y_test = test_labels

with h5.File(path2data + 'modelnet10_' + str(2 ** depth) + '.h5', 'w') as hf:
    hf.create_dataset('X_train', data=X_train.transpose(0,2,1))
    hf.create_dataset('y_train', data=y_train.transpose(0,2,1))
    hf.create_dataset('X_test', data=X_test.transpose(0,2,1))
    hf.create_dataset('y_test', data=y_test.transpose(0,2,1))