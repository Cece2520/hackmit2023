import numpy as np

def find_center(pose):
    # left shoulder, right shoulder, left hip, right hip
    return (pose[0,0,5,0:2] + pose[0,0,6,0:2] + pose[0,0,11,0:2] + pose[0,0,12,0:2])/4

def find_scale_factor(ref_pose, scored_pose, center):
    # left shoulder, right shoulder, left hip, right hip
    ref_points = [ref_pose[0,0,5,0:2], ref_pose[0,0,6,0:2], ref_pose[0,0,11,0:2], ref_pose[0,0,12,0:2]]
    scored_points = [scored_pose[0,0,5,0:2], scored_pose[0,0,6,0:2], scored_pose[0,0,11,0:2], scored_pose[0,0,12,0:2]]

    ref_dists = np.sqrt(np.sum((ref_points - center)**2, axis=1)) + 0.01
    scored_dists = np.sqrt(np.sum((scored_points - center)**2, axis=1)) + 0.01

    return np.sum(ref_dists/scored_dists)/len(ref_points)


def scale(pose, value):
    points = pose[:,:,:,0:2] / value
    return np.concatenate([points, pose[:,:,:,2:3]], axis=3)

def translate(pose, shift):
    points = pose[:,:,:,0:2] + shift
    return np.concatenate([points, pose[:,:,:,2:3]], axis=3)

def normalize(ref_pose, scored_pose):
    ref_center = find_center(ref_pose)
    scored_center = find_center(scored_pose)

    scored_pose = translate(scored_pose, ref_center - scored_center)
    scale_value = find_scale_factor(ref_pose, scored_pose, ref_center)
    scored_pose = scale(scored_pose, scale_value)
    return scored_pose


WINDOW_SIZE = 3
WEIGHTS = np.array([0.7, 0.7, 0.7, 1, 1, 1, 1, 1.5, 1.5, 1.5, 1.5, 1, 1, 1.5, 1.5, 1.5, 1.5])
DIFF_WEIGHTS = np.ones((1,17))
CUTOFFS = [5, 10, 20, 40]

def score(ref_pose, scored_pose, diff_ref_pose, diff_scored_pose):
    dists = np.sum((ref_pose[0,0,:,0:2] - scored_pose[0,0,:,0:2])**2, axis=1)
    dist_score = WEIGHTS @ dists

    diff_dists = np.sqrt(np.sum((diff_ref_pose[0,0,:,0:2] - diff_scored_pose[0,0,:,0:2])**2, axis=1))
    diff_dists_score = DIFF_WEIGHTS @ diff_dists

    return (dist_score + diff_dists_score) * 100




