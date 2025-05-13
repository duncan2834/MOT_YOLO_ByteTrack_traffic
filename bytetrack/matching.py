import numpy as np
from scipy.optimize import linear_sum_assignment

from supervision.detection.utils import box_iou_batch


def indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = matched_cost <= thresh

    matches = indices[matched_mask]
    # Tìm chỉ số những hàng (track) và cột (detection) không nằm trong bất kỳ cặp nào sau lọc
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0])) # 
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))
    return matches, unmatched_a, unmatched_b # a la track, b la detection

def iou_distance(atracks, btracks): # kcach cang nho, do trung hop cang cao
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
        len(btracks) > 0 and isinstance(btracks[0], np.ndarray)
    ):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]

    _ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    if _ious.size != 0:
        _ious = box_iou_batch(np.asarray(atlbrs), np.asarray(btlbrs)) # ham dung tinh IOU
    cost_matrix = 1 - _ious

    return cost_matrix
    
def linear_assignment(cost_matrix, thresh): 
    # Tìm ra các cặp (i, j) sao cho tổng chi phí cost_matrix[i, j] là nhỏ nhất, và loại bỏ những cặp có chi phí vượt ngưỡng thresh.
    # hungarian algo, thực hiện gán tối ưu giữa hai tập hợp (thường là các track và detection)
    if cost_matrix.size == 0: # ko co track or detection nao
        return (
            np.empty((0, 2), dtype=int),
            tuple(range(cost_matrix.shape[0])),
            tuple(range(cost_matrix.shape[1])),
        )
    cost_matrix[cost_matrix > thresh] = thresh + 1e-4 # Những phần tử chi phí vượt ngưỡng → bị gán giá trị lớn hơn để tránh bị chọn bởi thuật toán gán.
    # dung hungarian algorithm de chon cap (i, j) toi uu
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    indices = np.column_stack((row_ind, col_ind)) # cac cap (i, j) toi uu
    
    return indices_to_matches(cost_matrix, indices, thresh)

def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections]) # lay confidence score
    #  Biến det_scores thành ma trận có kích thước (M, N) bằng cách lặp lại theo số track (hàng) 
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    # là ma trận chi phí được điều chỉnh bởi confidence score
    return fuse_cost 