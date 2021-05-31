import tensorflow as tf
from db.utils import print_bboxes_names
from models.nms.nms import soft_nms, soft_nms_merge
from datetime import datetime
import numpy as np
import cv2
import copy
import json
import time
import os

def compute_result(top_bboxes, db_raw, categories, TB_obj=None, TB_iter=0, out_dir='./output'):

    if not os.path.exists(out_dir + '/results/'):
        os.makedirs(out_dir + '/results/')
    result_json = out_dir + '/results/' + str(TB_iter) + "-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".json"
    detections = db_raw.convert_to_coco(top_bboxes)
    with open(result_json, "w") as f:
        json.dump(detections, f)

    cls_ids = list(range(1, categories + 1))

    image_ids = list(top_bboxes.keys())

    db_raw.evaluate(result_json, cls_ids, image_ids, TB_obj=TB_obj, TB_iter=TB_iter)


def _nms(heat, kernel=1):
    hmax = tf.nn.max_pool(heat, ksize=kernel, strides=(1, 1), data_format="NCHW", padding='SAME')
    keep = tf.cast(tf.equal(hmax, heat), tf.float32)
    return heat * keep


def _rescale_dets(detections, ratios, borders, sizes):
    out = np.copy(detections)
    xs, ys = out[..., 0:4:2], out[..., 1:4:2]
    xs /= ratios[:, 1][:, None, None]
    ys /= ratios[:, 0][:, None, None]
    xs -= borders[:, 2][:, None, None]
    ys -= borders[:, 0][:, None, None]
    tx_inds = xs[:, :, 0] <= -5
    bx_inds = xs[:, :, 1] >= sizes[0, 1] + 5
    ty_inds = ys[:, :, 0] <= -5
    by_inds = ys[:, :, 1] >= sizes[0, 0] + 5

    np.clip(xs, 0, sizes[:, 1][:, None, None], out=xs)
    np.clip(ys, 0, sizes[:, 0][:, None, None], out=ys)
    out[:, tx_inds[0, :], 4] = -1
    out[:, bx_inds[0, :], 4] = -1
    out[:, ty_inds[0, :], 4] = -1
    out[:, by_inds[0, :], 4] = -1
    return out


def _rescale_cnts(center, ratios, borders, sizes):
    out = np.copy(center)
    out[..., [0]] /= ratios[:, 1][:, None, None]
    out[..., [1]] /= ratios[:, 0][:, None, None]
    out[..., [0]] -= borders[:, 2][:, None, None]
    out[..., [1]] -= borders[:, 0][:, None, None]
    np.clip(out[..., [0]], 0, sizes[:, 1][:, None, None], out=out[..., [0]])
    np.clip(out[..., [1]], 0, sizes[:, 0][:, None, None], out=out[..., [1]])
    return out


def process_output(input_ids, border, out_shape, pred_bboxes, pred_centers, db_raw, test_mod=2, printFlag=False,
                   name_base='./output/heatmaps/', TB_obj=None, TB_iter=0):
    K = db_raw.configs["top_k"]
    ae_threshold = db_raw.configs["ae_threshold"]
    nms_kernel = db_raw.configs["nms_kernel"]
    scales = db_raw.configs["test_scales"]
    weight_exp = db_raw.configs["weight_exp"]
    merge_bbox = db_raw.configs["merge_bbox"]
    categories = db_raw.configs["categories"]
    nms_threshold = db_raw.configs["nms_threshold"]
    max_per_image = db_raw.configs["max_per_image"]
    nms_algorithm = {"nms": 0, "linear_soft_nms": 1, "exp_soft_nms": 2}[db_raw.configs["nms_algorithm"]]

    ratios = np.zeros((1, 2), dtype=np.float32)
    sizes = np.zeros((1, 2), dtype=np.float32)
    borders = np.zeros((1, 4), dtype=np.float32)
    idx = 0

    top_bboxes = {}

    image_id = np.array(input_ids[idx])
    image_id_1 = ('COCO_train2014_{0:012d}').format(image_id) + '.' + db_raw._image_ids[0].split('.')[-1]
    image_id_2 = ('COCO_val2014_{0:012d}').format(image_id) + '.' + db_raw._image_ids[0].split('.')[-1]
    if image_id_1 in db_raw._image_ids:
        image_id = image_id_1
    elif image_id_2 in db_raw._image_ids:
        image_id = image_id_2
    else:
        print("ERROR - image id not in the dataset")

    image_file = db_raw._image_file.format(image_id)
    image_org = cv2.imread(image_file)

    start = time.time()

    height, width = image_org.shape[0:2]
    if test_mod == 0:
        inp_height = height
        inp_width = width
        border = np.array([0., height, 0., width])
    elif test_mod == 1:
        maxDim = np.argmax([height, width])
        if maxDim == 0:
            inp_height = height
            inp_width = height
            border[0] = border[0] * (height / 511)
        else:
            inp_height = width
            inp_width = width
            border[0] = border[0] * (width / 511)
    elif test_mod == 2:
        inp_height = height | 127
        inp_width = width | 127
    elif test_mod == 3:
        minDim = np.argmin([height, width])
        if minDim == 0:
            scale = height / out_shape[0]
        else:
            scale = width / out_shape[1]
        inp_height = int(float(out_shape[0]) * scale)
        inp_width = int(float(out_shape[1]) * scale)
        border[0] = np.array([0., height, 0., width])

    sizes[0] = [height, width]
    out_height, out_width = out_shape
    height_ratio = out_height / inp_height
    width_ratio = out_width / inp_width
    ratios[0] = [height_ratio, width_ratio]
    borders[0] = border[0]

    # gt_heatmap_tl = np.array(inputs[4][idx])
    # gt_heatmap_br = np.array(inputs[5][idx])
    # gt_heatmap_ct = np.array(inputs[6][idx])
    # # with decode=True, raw heatmaps
    # pred_heatmap_tl = np.array(tl_heat[idx])
    # pred_heatmap_br = np.array(br_heat[idx])
    # pred_heatmap_ct = np.array(ct_heat[idx])

    # print heatmaps - img, gt, pred
    # print_img_gt_pred_heatmaps(image, pred_heatmap_tl, gt_heatmap_tl,  name=name_base + str(image_id[:-4]) + '-tl-')
    # print_img_gt_pred_heatmaps(image, pred_heatmap_br, gt_heatmap_br,  name=name_base + str(image_id[:-4]) + '-br-')
    # print_img_gt_pred_heatmaps(image, pred_heatmap_ct, gt_heatmap_ct,  name=name_base + str(image_id[:-4]) + '-ct-')

    pred_bbox = np.expand_dims(np.array(pred_bboxes[idx]), axis=0)
    pred_bbox = _rescale_dets(pred_bbox, ratios, borders, sizes)[0]
    # print_bboxes(image_org, pred_bbox, name=name_base + str(image_id[:-4]) + '-pred_raw.png')

    pred_center = np.expand_dims(np.array(pred_centers[idx]), axis=0)
    pred_center = _rescale_cnts(pred_center, ratios, borders, sizes)[0]
    pred_bbox, pred_class = process_cnt(pred_bbox, pred_center)
    # print_bboxes(image_org, pred_bbox, name=name_base + str(image_id[:-4]) + '-pred_ct_filtered.png')

    end = time.time()

    top_bboxes[image_id] = top_bboxes_dic(pred_bbox, pred_class, categories, nms_threshold, nms_algorithm, weight_exp,
                                          merge_bbox, max_per_image)

    # printing bboxes
    pred_bbox_f = np.empty((0, 5), dtype=np.float32)
    pred_bbox_f_names = []
    for key, val in top_bboxes[image_id].items():
        if len(val) > 0:
            pred_bbox_f = np.concatenate((pred_bbox_f, val), axis=0)
            pred_bbox_f_names += [[key, db_raw.class_name(key)]] * len(val)
    if printFlag:
        out_img = print_bboxes_names(image_org, pred_bbox_f, pred_bbox_f_names, thresh=0.4,
                                     name=name_base + str(image_id[:-4]) + '-pred_nms_filtered-' + str(
                                         test_mod) + '.png')
        TB_obj.log_image('validation_images', image_id, np.expand_dims(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB), 0),
                         TB_iter)
    return top_bboxes, end - start


def process_cnt(detections, center_points):
    valid_ind = detections[:, 4] > -1
    valid_detections = detections[valid_ind]

    box_width = valid_detections[:, 2] - valid_detections[:, 0]
    box_height = valid_detections[:, 3] - valid_detections[:, 1]

    s_ind = (box_width * box_height <= 22500)
    l_ind = (box_width * box_height > 22500)

    s_detections = valid_detections[s_ind]
    l_detections = valid_detections[l_ind]

    s_left_x = (2 * s_detections[:, 0] + s_detections[:, 2]) / 3
    s_right_x = (s_detections[:, 0] + 2 * s_detections[:, 2]) / 3
    s_top_y = (2 * s_detections[:, 1] + s_detections[:, 3]) / 3
    s_bottom_y = (s_detections[:, 1] + 2 * s_detections[:, 3]) / 3

    s_temp_score = copy.copy(s_detections[:, 4])
    s_detections[:, 4] = -1

    center_x = center_points[:, 0][:, np.newaxis]
    center_y = center_points[:, 1][:, np.newaxis]
    s_left_x = s_left_x[np.newaxis, :]
    s_right_x = s_right_x[np.newaxis, :]
    s_top_y = s_top_y[np.newaxis, :]
    s_bottom_y = s_bottom_y[np.newaxis, :]

    ind_lx = (center_x - s_left_x) > 0
    ind_rx = (center_x - s_right_x) < 0
    ind_ty = (center_y - s_top_y) > 0
    ind_by = (center_y - s_bottom_y) < 0
    ind_cls = (center_points[:, 2][:, np.newaxis] - s_detections[:, -1][np.newaxis, :]) == 0
    ind_s_new_score = np.max(((ind_lx + 0) & (ind_rx + 0) & (ind_ty + 0) & (ind_by + 0) & (ind_cls + 0)), axis=0) == 1
    index_s_new_score = np.argmax(
        ((ind_lx + 0) & (ind_rx + 0) & (ind_ty + 0) & (ind_by + 0) & (ind_cls + 0))[:, ind_s_new_score], axis=0)
    s_detections[:, 4][ind_s_new_score] = (s_temp_score[ind_s_new_score] * 2 + center_points[
        index_s_new_score, 3]) / 3

    l_left_x = (3 * l_detections[:, 0] + 2 * l_detections[:, 2]) / 5
    l_right_x = (2 * l_detections[:, 0] + 3 * l_detections[:, 2]) / 5
    l_top_y = (3 * l_detections[:, 1] + 2 * l_detections[:, 3]) / 5
    l_bottom_y = (2 * l_detections[:, 1] + 3 * l_detections[:, 3]) / 5

    l_temp_score = copy.copy(l_detections[:, 4])
    l_detections[:, 4] = -1

    center_x = center_points[:, 0][:, np.newaxis]
    center_y = center_points[:, 1][:, np.newaxis]
    l_left_x = l_left_x[np.newaxis, :]
    l_right_x = l_right_x[np.newaxis, :]
    l_top_y = l_top_y[np.newaxis, :]
    l_bottom_y = l_bottom_y[np.newaxis, :]

    ind_lx = (center_x - l_left_x) > 0
    ind_rx = (center_x - l_right_x) < 0
    ind_ty = (center_y - l_top_y) > 0
    ind_by = (center_y - l_bottom_y) < 0
    ind_cls = (center_points[:, 2][:, np.newaxis] - l_detections[:, -1][np.newaxis, :]) == 0
    ind_l_new_score = np.max(((ind_lx + 0) & (ind_rx + 0) & (ind_ty + 0) & (ind_by + 0) & (ind_cls + 0)), axis=0) == 1
    index_l_new_score = np.argmax(
        ((ind_lx + 0) & (ind_rx + 0) & (ind_ty + 0) & (ind_by + 0) & (ind_cls + 0))[:, ind_l_new_score], axis=0)
    l_detections[:, 4][ind_l_new_score] = (l_temp_score[ind_l_new_score] * 2 + center_points[
        index_l_new_score, 3]) / 3

    detections = np.concatenate([l_detections, s_detections], axis=0)
    detections = detections[np.argsort(-detections[:, 4])]

    classes = detections[..., -1]
    keep_inds = (detections[:, 4] > -1)

    detections = detections[keep_inds]
    classes = classes[keep_inds]
    return detections, classes


def top_bboxes_dic(detections, classes, categories, nms_threshold, nms_algorithm, weight_exp, merge_bbox,
                   max_per_image):
    out = {}
    for j in range(categories):
        keep_inds = (classes == j)
        out[j + 1] = detections[keep_inds][:, 0:7].astype(np.float32)
        if merge_bbox:
            soft_nms_merge(out[j + 1], Nt=nms_threshold, method=nms_algorithm, weight_exp=weight_exp)
        else:
            soft_nms(out[j + 1], Nt=nms_threshold, method=nms_algorithm)
        out[j + 1] = out[j + 1][:, 0:5]

    scores = np.hstack([out[j][:, -1] for j in range(1, categories + 1)])

    if len(scores) > max_per_image:
        kth = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, categories + 1):
            keep_inds = (out[j][:, -1] >= thresh)
            out[j] = out[j][keep_inds]
    return out
