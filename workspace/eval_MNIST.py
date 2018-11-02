import os
import glob
import numpy as np
import tqdm
import time
import datetime
from pickle import load, dump
from open3d import PointCloud, Feature, Vector3dVector, compute_fpfh_feature, KDTreeSearchParamHybrid, \
    estimate_normals, voxel_down_sample, registration_icp, registration_fast_based_on_feature_matching, \
    ICPConvergenceCriteria, TransformationEstimationPointToPoint, FastGlobalRegistrationOption

from make_eval_set import to_ply, Store


def predict_icp(source, target, trans_init=np.identity(4), threshold=1.0, max_iter=200):
    reg_p2p = registration_icp(source, target, threshold, trans_init,
                               TransformationEstimationPointToPoint(),
                               ICPConvergenceCriteria(max_iteration=max_iter))
    return reg_p2p.transformation


def predict_fgr(source, target, source_features, target_features, voxel_size=0.01):
    radius_normal = voxel_size * 2
    estimate_normals(source, KDTreeSearchParamHybrid(
        radius=radius_normal, max_nn=30))
    estimate_normals(target, KDTreeSearchParamHybrid(
        radius=radius_normal, max_nn=30))

    distance_threshold = voxel_size * 0.5
    result = registration_fast_based_on_feature_matching(
        source, target, source_features, target_features,
        FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result.transformation


def predict_all(path):
    batch = load(open(path, 'rb'))
    T_preds = []
    times = []

    source_pc = to_ply(batch.source_KD_cloud)
    target_pc = to_ply(batch.target_KD_cloud)

    source_pc_downsampled = to_ply(batch.source_FPFH_cloud)
    target_pc_downsampled = to_ply(batch.target_FPFH_cloud)

    source_features = Feature()
    target_features = Feature()

    # ICP with coords only
    start = time.time()
    T_preds.append(predict_icp(source_pc, target_pc))
    times.append(time.time() - start)

    # FGR with coords only
    source_features.data = batch.source_FPFH_cloud.T
    target_features.data = batch.target_FPFH_cloud.T

    start = time.time()
    T_preds.append(predict_fgr(source_pc_downsampled, target_pc_downsampled,
                               source_features, target_features))
    times.append(time.time() - start)

    # FGR with FPFH
    source_features.data = batch.source_FPFH_features
    target_features.data = batch.target_FPFH_features

    start = time.time()
    T_preds.append(predict_fgr(source_pc_downsampled, target_pc_downsampled,
                               source_features, target_features))
    times.append(time.time() - start)

    # FGR with KD features
    for source_KD_features, target_KD_features in zip(batch.source_KD_features, batch.target_KD_features):
        source_pc_downsampled = to_ply(source_KD_features[:, :3])
        target_pc_downsampled = to_ply(target_KD_features[:, :3])

        source_features.data = source_KD_features[:, 3:].T
        target_features.data = target_KD_features[:, 3:].T

        start = time.time()
        T_preds.append(predict_fgr(source_pc_downsampled, target_pc_downsampled,
                                   source_features, target_features))
        times.append(time.time() - start)

    return np.array(T_preds), np.array(times)


if __name__ == "__main__":
    paths = glob.glob("../datasets/MNIST_2D/*/*/*.pkl")
    result = []
    for p in tqdm.tqdm(paths):
        result.append(predict_all(p))

    T_preds = np.array([x[0] for x in result])
    times = np.array([x[1] for x in result])

    path = '../results/MNIST_2D/raw/'
    file_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S.results.pkl")
    dump((paths, T_preds, times), open(os.path.join(path, file_name), 'wb+'))
