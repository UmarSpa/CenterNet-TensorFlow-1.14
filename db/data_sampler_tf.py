import numpy as np
import tensorflow as tf
import time
from .utils import draw_gaussian, gaussian_radius, random_h_flip, clip_detections, normalize_image, resize_image, \
    random_crop, full_image_crop, color_jittering, lighting_, resize_image_keepratio, _crop_image


def tf_dataset_init(db, sys_cfg):
    db_cfg = db.configs
    filtered_ids, filtered_ids_nums, dec_samples, dec_lens, dec_ids = [], [], [], [], []

    print(" - Samples before filtering: {}".format(db.db_inds.size))

    for img_id in db._image_ids:
        dec = db._detections[img_id]
        if len(dec) > 0:
            filtered_ids.append(img_id)
            dec_samples.append(dec[:, 0:4])
            dec_lens.append(len(dec))
            dec_ids.append(dec[:, 4])
            filtered_ids_nums.append(int(img_id.split('_')[-1].split('.')[0]))

    dec_samples_np = np.zeros((len(dec_lens), max(dec_lens), 4), np.float32)
    dec_ids_np = np.zeros((len(dec_lens), max(dec_lens)), np.float32)
    for x_id in range(len(dec_lens)):
        dec_samples_np[x_id, 0:dec_lens[x_id], :] = dec_samples[x_id]
        dec_ids_np[x_id, 0:dec_lens[x_id]] = dec_ids[x_id]

    print(" - Samples after filtering: {}".format(len(filtered_ids)))

    if db._split == 'minival':
        if sys_cfg['debug_flag']:
            filtered_ids = filtered_ids[:100]
            filtered_ids_nums = filtered_ids_nums[:100]
            dec_samples_np = dec_samples_np[:100]
            dec_lens = dec_lens[:100]
            dec_ids_np = dec_ids_np[:100]

    tf_dataset = tf.data.Dataset.from_tensor_slices(
        (filtered_ids, filtered_ids_nums, dec_samples_np, dec_lens, dec_ids_np))

    if db._split == 'trainval':
        tf_dataset = tf_dataset.shuffle(len(filtered_ids))

    start = time.time()
    tf_dataset = tf_dataset.map(
        lambda img_id, img_id_num, dec_sample, dec_len, dec_id: _parse_function(img_id, img_id_num, dec_sample, dec_len,
                                                                                dec_id, db._mean,
                                                                                db._std, db._eig_val, db._eig_vec,
                                                                                db._image_dir + "/",
                                                                                db_cfg['input_size'],
                                                                                db_cfg['output_sizes'][0],
                                                                                db_cfg['gaussian_bump'],
                                                                                db_cfg['gaussian_radius'],
                                                                                db_cfg['gaussian_iou'],
                                                                                db_cfg['border'], db_cfg['rand_crop'],
                                                                                db_cfg['rand_scales'],
                                                                                db_cfg['rand_color'],
                                                                                db_cfg['lighting'],
                                                                                db_cfg['categories'],
                                                                                train_flag=(db._split == 'trainval'),
                                                                                input_mod=db_cfg['input_mod']),
        num_parallel_calls=4
    )
    print(" - input modality: {}".format(db_cfg['input_mod']))
    print(" - tf dataset creation time: {0:2.2f} sec".format(time.time() - start))

    if db._split == 'trainval':
        tf_dataset = tf_dataset.repeat(-1).batch(sys_cfg['batch_size'], drop_remainder=True).prefetch(
            sys_cfg['prefetch_size'])

    else:
        numEle = len(filtered_ids)
        tf_dataset = tf_dataset.take(numEle).batch(1, drop_remainder=True).prefetch(sys_cfg['prefetch_size'])

    return tf_dataset


def _parse_function(img_id, img_id_num, dec_sample, dec_len, dec_id, mean, std, eig_val, eig_vec, image_dir, input_size,
                    output_size, gauss_bump, gauss_rad, gauss_iou, border, random_crop_flag, random_scales, rand_color,
                    lighting, cats, train_flag=False, input_mod=0):
    input_size = tf.cast(input_size, dtype=tf.float32)
    output_size = tf.cast(output_size, dtype=tf.float32)
    mean = tf.cast(mean, dtype=tf.float32)
    std = tf.cast(std, dtype=tf.float32)
    max_tag_len = tf.constant(128, dtype=tf.int32)

    image_file = tf.read_file(image_dir + img_id)
    image_ = tf.image.decode_jpeg(image_file, channels=3, dct_method='INTEGER_ACCURATE')
    image = tf.reverse(image_, axis=[2])
    detections = tf.concat([dec_sample[0:dec_len], dec_id[:dec_len, tf.newaxis]], axis=1)

    if train_flag:
        if random_crop_flag:
            img, detects = random_crop(image, detections, random_scales, input_size, border=border)
            img, detects = resize_image(img, detects, input_size)
            detects = clip_detections(img, detects)
            image, detections = tf.cond(tf.equal(tf.shape(detects)[0], 0),
                                        lambda: (tf.cast(image, tf.float32), detections), lambda: (img, detects))
        else:
            image, detections, _ = full_image_crop(image, detections)

        image, detections = resize_image(image, detections, input_size)

    else:
        if input_mod == 0:  # fixed input size of 511x511
            image, detections = resize_image(image, detections, input_size)

        elif input_mod == 1:  # fixed input size of 511x511, keeping the aspect ration
            image, detections, border = full_image_crop(image, detections)

            h_ratio = tf.div(input_size[0], tf.cast(tf.shape(image)[0], tf.float32))
            w_ratio = tf.div(input_size[1], tf.cast(tf.shape(image)[1], tf.float32))
            border[0] = tf.floor(tf.multiply(border[0], h_ratio))
            border[1] = tf.floor(tf.multiply(border[1], h_ratio))
            border[2] = tf.floor(tf.multiply(border[2], w_ratio))
            border[3] = tf.floor(tf.multiply(border[3], w_ratio))

            image, detections = resize_image(image, detections, input_size)

        elif input_mod == 2:  # original input size with border
            height, width = tf.shape(image)[0], tf.shape(image)[1]
            new_center = [tf.math.floordiv(tf.cast(height, tf.float32), 2.),
                          tf.math.floordiv(tf.cast(width, tf.float32), 2.)]  # np.array([height // 2, width // 2])

            inp_height = tf.cast(tf.bitwise.bitwise_or(height, 127), tf.float32)  # height | 127
            inp_width = tf.cast(tf.bitwise.bitwise_or(width, 127), tf.float32)  # width | 127

            image, border, offset = _crop_image(image, new_center, [inp_height, inp_width])

            width_detect = tf.add(detections[:, 0:4:2], border[2])
            height_detect = tf.add(detections[:, 1:4:2], border[0])
            detections_ = tf.reshape(
                tf.concat([width_detect[..., tf.newaxis], height_detect[..., tf.newaxis]], axis=-1),
                [tf.shape(width_detect)[0], -1])
            detections = tf.concat([detections_, detections[:, 4, tf.newaxis]], axis=1)

        elif input_mod == 3:  # rescale to keep min dimension of size 511.
            image, detections, border = resize_image_keepratio(image, detections,
                                                               tf.convert_to_tensor([511, 2000], dtype=tf.int32))

    detections = clip_detections(image, detections)

    if train_flag:
        image, detections = random_h_flip(image, detections)

    image = tf.div(tf.cast(image, tf.float32), 255.)

    if train_flag:
        if rand_color:
            image = color_jittering(image)
            if lighting:
                image = lighting_(image, 0.1, eig_val, eig_vec)

    image = normalize_image(image, mean, std)

    image = tf.transpose(image, perm=[2, 0, 1])

    if not train_flag:
        return image, img_id_num, border

    width_ratio = tf.div(output_size[1], tf.cast(tf.shape(image)[2], tf.float32))
    height_ratio = tf.div(output_size[0], tf.cast(tf.shape(image)[1], tf.float32))

    tl_tag, br_tag, ct_tag, tl_heatmap, br_heatmap, ct_heatmap, tag_mask, tl_regr, br_regr, ct_regr = _bbs_processing(
        detections, output_size, gauss_bump, gauss_rad, gauss_iou, cats, width_ratio, height_ratio, max_tag_len)

    return image, tl_tag, br_tag, ct_tag, tl_heatmap, br_heatmap, ct_heatmap, tag_mask, tl_regr, br_regr, ct_regr, img_id_num, border


def _bbs_processing(detections, output_size, gauss_bump, gauss_rad, gauss_iou, cats, width_ratio, height_ratio,
                    max_tag_len):
    tl_regr = tf.zeros((max_tag_len, 2), dtype=tf.float32)
    br_regr = tf.zeros((max_tag_len, 2), dtype=tf.float32)
    ct_regr = tf.zeros((max_tag_len, 2), dtype=tf.float32)
    tl_tag = tf.zeros((max_tag_len), dtype=tf.int64)
    br_tag = tf.zeros((max_tag_len), dtype=tf.int64)
    ct_tag = tf.zeros((max_tag_len), dtype=tf.int64)
    tag_mask = tf.zeros((max_tag_len), dtype=tf.int8)

    detection_out = tf.map_fn(
        lambda x: _heatmap_generation(x, output_size, gauss_bump, gauss_rad, gauss_iou, cats, width_ratio,
                                      height_ratio),
        detections,
        dtype=(
            tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.int64, tf.int64, tf.int64,
            tf.int32))

    dec_range = tf.range(tf.shape(detections)[0])[..., tf.newaxis]

    tl_heatmap = tf.reduce_max(detection_out[0], axis=0)
    br_heatmap = tf.reduce_max(detection_out[1], axis=0)
    ct_heatmap = tf.reduce_max(detection_out[2], axis=0)
    tl_regr = tf.tensor_scatter_update(tl_regr, dec_range, detection_out[3])
    br_regr = tf.tensor_scatter_update(br_regr, dec_range, detection_out[4])
    ct_regr = tf.tensor_scatter_update(ct_regr, dec_range, detection_out[5])
    tl_tag = tf.tensor_scatter_update(tl_tag, dec_range, detection_out[6])
    br_tag = tf.tensor_scatter_update(br_tag, dec_range, detection_out[7])
    ct_tag = tf.tensor_scatter_update(ct_tag, dec_range, detection_out[8])
    tag_mask = tf.tensor_scatter_update(tag_mask, dec_range,
                                        tf.cast(tf.ones_like(tf.squeeze(dec_range, axis=-1)), tf.int8))

    return tl_tag, br_tag, ct_tag, tl_heatmap, br_heatmap, ct_heatmap, tag_mask, tl_regr, br_regr, ct_regr


def _heatmap_generation(detection, output_size, gauss_bump, gauss_rad, gauss_iou, cats, width_ratio, height_ratio):
    output_size = tf.cast(output_size, tf.float32)

    tl_heatmap = tf.zeros((cats, output_size[0], output_size[1]), dtype=tf.float32)
    br_heatmap = tf.zeros((cats, output_size[0], output_size[1]), dtype=tf.float32)
    ct_heatmap = tf.zeros((cats, output_size[0], output_size[1]), dtype=tf.float32)

    category = tf.cast((detection[4]) - 1, tf.int32)

    xtl, ytl = detection[0], detection[1]
    xbr, ybr = detection[2], detection[3]
    xct = tf.div(tf.add(detection[2], detection[0]), 2.)
    yct = tf.div(tf.add(detection[3], detection[1]), 2.)

    fxtl = tf.multiply(xtl, width_ratio)
    fytl = tf.multiply(ytl, height_ratio)
    fxbr = tf.multiply(xbr, width_ratio)
    fybr = tf.multiply(ybr, height_ratio)
    fxct = tf.multiply(xct, width_ratio)
    fyct = tf.multiply(yct, height_ratio)

    xtl = tf.cast(tf.cast(fxtl, tf.int32), tf.float32)
    ytl = tf.cast(tf.cast(fytl, tf.int32), tf.float32)
    xbr = tf.cast(tf.cast(fxbr, tf.int32), tf.float32)
    ybr = tf.cast(tf.cast(fybr, tf.int32), tf.float32)
    xct = tf.cast(tf.cast(fxct, tf.int32), tf.float32)
    yct = tf.cast(tf.cast(fyct, tf.int32), tf.float32)

    if gauss_bump:
        width = tf.subtract(detection[2], detection[0])
        height = tf.subtract(detection[3], detection[1])

        width = tf.ceil(tf.multiply(width, width_ratio))
        height = tf.ceil(tf.multiply(height, height_ratio))

        if gauss_rad == -1:
            radius = gaussian_radius((height, width), gauss_iou)
            radius = tf.maximum(0, tf.cast(radius, tf.int32))
        else:
            radius = gauss_rad

        heatmap_ = draw_gaussian(tl_heatmap[category], [xtl, ytl], radius)
        tl_heatmap = tf.tensor_scatter_update(tl_heatmap, category[tf.newaxis, tf.newaxis], heatmap_[tf.newaxis])
        heatmap_ = draw_gaussian(br_heatmap[category], [xbr, ybr], radius)
        br_heatmap = tf.tensor_scatter_update(br_heatmap, category[tf.newaxis, tf.newaxis], heatmap_[tf.newaxis])
        heatmap_ = draw_gaussian(ct_heatmap[category], [xct, yct], radius, delte=5.)
        ct_heatmap = tf.tensor_scatter_update(ct_heatmap, category[tf.newaxis, tf.newaxis], heatmap_[tf.newaxis])
    else:
        # verify if this part is working
        tl_heatmap = tf.tensor_scatter_update(tl_heatmap, tf.stack([category, ytl, xtl], axis=0)[tf.newaxis], [1.])
        br_heatmap = tf.tensor_scatter_update(br_heatmap, tf.stack([category, ybr, xbr], axis=0)[tf.newaxis], [1.])
        ct_heatmap = tf.tensor_scatter_update(ct_heatmap, tf.stack([category, yct, xct], axis=0)[tf.newaxis], [1.])

    tl_regr = tf.stack([tf.subtract(fxtl, xtl), tf.subtract(fytl, ytl)], axis=0)
    br_regr = tf.stack([tf.subtract(fxbr, xbr), tf.subtract(fybr, ybr)], axis=0)
    ct_regr = tf.stack([tf.subtract(fxct, xct), tf.subtract(fyct, yct)], axis=0)
    tl_tag = tf.cast(tf.add(tf.multiply(ytl, output_size[1]), xtl), tf.int64)
    br_tag = tf.cast(tf.add(tf.multiply(ybr, output_size[1]), xbr), tf.int64)
    ct_tag = tf.cast(tf.add(tf.multiply(yct, output_size[1]), xct), tf.int64)

    return tl_heatmap, br_heatmap, ct_heatmap, tl_regr, br_regr, ct_regr, tl_tag, br_tag, ct_tag, category
