import tensorflow as tf
import numpy as np
import cv2

COLORS = np.load('./db/COLORS.npy')


def gaussian2D(shape, sigma=1.0):
    m = tf.div(tf.subtract(shape[0], 1.), 2.)
    n = tf.div(tf.subtract(shape[1], 1.), 2.)
    y = tf.range(-m, tf.add(m, 1))[..., tf.newaxis]
    x = tf.range(-n, tf.add(n, 1))[tf.newaxis, ...]
    h = tf.exp(- tf.div(tf.add(tf.multiply(x, x), tf.multiply(y, y)), tf.multiply(2., tf.multiply(sigma, sigma))))
    h = tf.where(tf.less(h, tf.multiply(1.2e-07, tf.reduce_max(h))), tf.zeros_like(h), h)
    return h


def draw_gaussian(heatmap, center, radius, k=1., delte=6.):
    radius = tf.cast(radius, tf.float32)
    center = tf.cast(center, tf.float32)

    diameter = tf.add(tf.multiply(2., radius), 1.)
    gaussian = gaussian2D((diameter, diameter), sigma=tf.div(diameter, tf.cast(delte, tf.float32)))

    x, y = center[0], center[1]
    height, width = tf.shape(heatmap)[0], tf.shape(heatmap)[1]

    left, right = tf.minimum(x, radius), tf.minimum(tf.subtract(tf.cast(width, tf.float32), x), tf.add(radius, 1.))
    top, bottom = tf.minimum(y, radius), tf.minimum(tf.subtract(tf.cast(height, tf.float32), y), tf.add(radius, 1.))

    masked_gaussian = tf.multiply(
        gaussian[tf.cast(tf.subtract(radius, top), tf.int32):tf.cast(tf.add(radius, bottom), tf.int32),
        tf.cast(tf.subtract(radius, left), tf.int32):tf.cast(tf.add(radius, right), tf.int32)], k)

    rows = tf.range(tf.cast(tf.subtract(y, top), tf.int32), tf.cast(tf.add(y, bottom), tf.int32))
    cols = tf.range(tf.cast(tf.subtract(x, left), tf.int32), tf.cast(tf.add(x, right), tf.int32))
    ii, jj = tf.meshgrid(cols, rows)
    heatmap = tf.tensor_scatter_update(heatmap, tf.stack([jj, ii], axis=-1), masked_gaussian)
    return heatmap


def gaussian_radius(det_size, min_overlap):
    height, width = det_size
    a1 = 1.
    b1 = tf.add(height, width)
    c1 = tf.div(tf.multiply(tf.multiply(width, height), tf.subtract(1., min_overlap)), tf.add(1., min_overlap))
    sq1 = tf.sqrt(tf.subtract(tf.multiply(b1, b1), tf.multiply(tf.multiply(4., a1), c1)))
    r1 = tf.div(tf.add(b1, sq1), 2.)

    a2 = 4.
    b2 = tf.multiply(2., tf.add(height, width))
    c2 = tf.multiply(tf.subtract(1., min_overlap), tf.multiply(width, height))
    sq2 = tf.sqrt(tf.subtract(tf.multiply(b2, b2), tf.multiply(4., tf.multiply(a2, c2))))
    r2 = tf.div(tf.add(b2, sq2), 2.)

    a3 = tf.multiply(4., min_overlap)
    b3 = tf.multiply(-2., tf.multiply(min_overlap, tf.add(height, width)))
    c3 = tf.multiply(tf.subtract(min_overlap, 1.), tf.multiply(width, height))
    sq3 = tf.sqrt(tf.subtract(tf.multiply(b3, b3), tf.multiply(4., tf.multiply(a3, c3))))
    r3 = tf.div(tf.add(b3, sq3), 2.)
    return tf.minimum(tf.minimum(r1, r2), r3)


def h_flip_detections(bbs, w):
    y1, x1, y2, x2 = tf.unstack(bbs, axis=1)
    w = tf.cast(w, tf.dtypes.float32)
    flipped = tf.stack([w - y2 - 1, x1, w - y1 - 1, x2], axis=1)
    return flipped


def _f1(img, bbs):
    img = tf.image.flip_left_right(img)
    bbs_ = h_flip_detections(tf.slice(bbs, [0, 0], [-1, 4]), tf.shape(img)[1])
    bbs = tf.concat([bbs_, bbs[:, 4, tf.newaxis]], axis=1)
    return img, bbs


def random_h_flip(image, detections):
    flip_flag = tf.random.uniform([], maxval=2, dtype=tf.int32)
    image, detections = tf.cond(tf.equal(flip_flag, 1), lambda: _f1(image, detections), lambda: (image, detections))
    return image, detections


def clip_detections(image, detections):
    height = tf.cast(tf.shape(image)[0], tf.float32)
    width = tf.cast(tf.shape(image)[1], tf.float32)

    width_detect = tf.clip_by_value(detections[:, 0:4:2], 0, width - 1)
    height_detect = tf.clip_by_value(detections[:, 1:4:2], 0, height - 1)

    detections_ = tf.reshape(tf.concat([width_detect[..., tf.newaxis], height_detect[..., tf.newaxis]], axis=-1),
                             [tf.shape(width_detect)[0], -1])
    detections = tf.concat([detections_, detections[:, 4, tf.newaxis]], axis=1)
    zeros_mask = tf.zeros_like(detections[:, 2])
    mask_1 = tf.greater(tf.subtract(detections[:, 2], detections[:, 0]), zeros_mask)
    mask_2 = tf.greater(tf.subtract(detections[:, 3], detections[:, 1]), zeros_mask)
    mask = tf.logical_and(mask_1, mask_2)
    detections_out = tf.boolean_mask(detections, mask)
    return detections_out


def normalize_image(x, mean, std):
    return tf.div(tf.subtract(x, mean), std)


def resize_image(image, detections, size):
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    new_height = size[0]
    new_width = size[1]

    image = tf.image.resize_images(image, (new_height, new_width))

    height_ratio = tf.div(new_height, tf.cast(height, tf.float32))
    width_ratio = tf.div(new_width, tf.cast(width, tf.float32))
    width_detect = tf.multiply(detections[:, 0:4:2], width_ratio)
    height_detect = tf.multiply(detections[:, 1:4:2], height_ratio)
    detections_ = tf.reshape(tf.concat([width_detect[..., tf.newaxis], height_detect[..., tf.newaxis]], axis=-1),
                             [tf.shape(width_detect)[0], -1])
    detections = tf.concat([detections_, detections[:, 4, tf.newaxis]], axis=1)
    return image, detections


def resize_image_keepratio(img, detections, scale):
    height, width = tf.shape(img)[0], tf.shape(img)[1]

    max_short_edge, max_long_edge = tf.math.reduce_min(scale), tf.math.reduce_max(scale)
    scale_factor = tf.math.reduce_min(
        tf.stack([
            max_long_edge / tf.math.reduce_max(tf.stack([tf.shape(img)[0], tf.shape(img)[1]])),
            max_short_edge / tf.math.reduce_min(tf.stack([tf.shape(img)[0], tf.shape(img)[1]]))
        ])
    )

    new_height = tf.cast(tf.cast(height, tf.float64) * scale_factor + 0.5, tf.int32)
    new_width = tf.cast(tf.cast(width, tf.float64) * scale_factor + 0.5, tf.int32)

    res_img = tf.image.resize(img, (new_height, new_width))

    rows = tf.range(0, new_height)
    cols = tf.range(0, new_width)
    ii, jj = tf.meshgrid(cols, rows)
    new_height_f = tf.cast(tf.bitwise.bitwise_or(new_height, 127), tf.int32)
    new_width_f = tf.cast(tf.bitwise.bitwise_or(new_width, 127), tf.int32)
    out_img = tf.zeros((new_height_f, new_width_f, 3), dtype=res_img.dtype)
    out_img = tf.tensor_scatter_update(out_img, tf.stack([jj, ii], axis=-1), res_img)

    height_ratio = tf.div(tf.cast(new_height, tf.float32), tf.cast(height, tf.float32))
    width_ratio = tf.div(tf.cast(new_width, tf.float32), tf.cast(width, tf.float32))
    width_detect = tf.multiply(detections[:, 0:4:2], width_ratio)
    height_detect = tf.multiply(detections[:, 1:4:2], height_ratio)
    detections_ = tf.reshape(tf.concat([width_detect[..., tf.newaxis], height_detect[..., tf.newaxis]], axis=-1),
                             [tf.shape(width_detect)[0], -1])
    detections = tf.concat([detections_, detections[:, 4, tf.newaxis]], axis=1)

    border = [tf.convert_to_tensor(0., dtype=tf.float32), tf.cast(new_height, tf.float32),
              tf.convert_to_tensor(0., dtype=tf.float32), tf.cast(new_width, tf.float32)]

    return out_img, detections, border


def bbox_rescale(bboxes, img_shape, scale_factor):
    bbx = bboxes[:, :4]
    bbx = bbx * tf.cast(scale_factor, tf.dtypes.float32)

    y1, x1, y2, x2 = tf.unstack(bbx, axis=1)
    img_shape = tf.cast(img_shape, tf.dtypes.float32)
    bbx = tf.stack([
        tf.clip_by_value(y1, 0.0, img_shape[0]),
        tf.clip_by_value(x1, 0.0, img_shape[1]),
        tf.clip_by_value(y2, 0.0, img_shape[0]),
        tf.clip_by_value(x2, 0.0, img_shape[1])
    ], axis=1)

    bboxes = tf.concat([bbx, bboxes[:, 4, tf.newaxis]], axis=1)

    return bboxes


def _body(border, size, i):
    return border, size, tf.multiply(i, 2)


def _condition(border, size, i):
    return tf.less_equal(tf.subtract(size, tf.floor_div(border, i)), tf.floor_div(border, i))


def _get_border(border, size):
    temp = tf.while_loop(_condition, _body, [border, size, 1.])
    return tf.floor_div(border, temp[-1])


def random_crop(image, detections, random_scales, view_size, border=64):
    view_height, view_width = view_size[0], view_size[1]
    image_height, image_width = tf.cast(tf.shape(image)[0], tf.float32), tf.cast(tf.shape(image)[1], tf.float32)

    scale = tf.random.shuffle(tf.cast(random_scales, tf.float32))[0]
    height = tf.cast(tf.cast(tf.multiply(view_height, scale), tf.int32), tf.float32)
    width = tf.cast(tf.cast(tf.multiply(view_width, scale), tf.int32), tf.float32)

    cropped_image = tf.zeros((height, width, 3), dtype=image.dtype)

    w_border = _get_border(tf.cast(border, tf.float32), image_width)
    h_border = _get_border(tf.cast(border, tf.float32), image_height)

    ctx = tf.cast(tf.random.uniform([], minval=tf.cast(w_border, tf.int32),
                                    maxval=tf.cast(tf.subtract(image_width, w_border), tf.int32), dtype=tf.int32),
                  dtype=tf.float32)
    cty = tf.cast(tf.random.uniform([], minval=tf.cast(h_border, tf.int32),
                                    maxval=tf.cast(tf.subtract(image_height, h_border), tf.int32), dtype=tf.int32),
                  dtype=tf.float32)

    x0, x1 = tf.maximum(tf.subtract(ctx, tf.floor_div(width, 2.)), 0.), tf.minimum(tf.add(ctx, tf.floor_div(width, 2.)),
                                                                                   image_width)
    y0, y1 = tf.maximum(tf.subtract(cty, tf.floor_div(height, 2.)), 0.), tf.minimum(
        tf.add(cty, tf.floor_div(height, 2.)), image_height)

    left_w, right_w = tf.subtract(ctx, x0), tf.subtract(x1, ctx)
    top_h, bottom_h = tf.subtract(cty, y0), tf.subtract(y1, cty)

    cropped_ctx, cropped_cty = tf.floor_div(width, 2.), tf.floor_div(height, 2.)

    sliced_image = tf.slice(image, [tf.cast(y0, tf.int32), tf.cast(x0, tf.int32), 0],
                            [tf.cast(tf.subtract(y1, y0), tf.int32), tf.cast(tf.subtract(x1, x0), tf.int32), 3])
    rows = tf.range(tf.cast(tf.subtract(cropped_cty, top_h), tf.int32),
                    tf.cast(tf.add(cropped_cty, bottom_h), tf.int32))
    cols = tf.range(tf.cast(tf.subtract(cropped_ctx, left_w), tf.int32),
                    tf.cast(tf.add(cropped_ctx, right_w), tf.int32))
    ii, jj = tf.meshgrid(cols, rows)

    cropped_image = tf.tensor_scatter_update(cropped_image, tf.stack([jj, ii], axis=-1), sliced_image)

    width_detect = tf.subtract(detections[:, 0:4:2], x0)
    height_detect = tf.subtract(detections[:, 1:4:2], y0)
    width_detect = tf.add(width_detect, tf.subtract(cropped_ctx, left_w))
    height_detect = tf.add(height_detect, tf.subtract(cropped_cty, top_h))

    detections_ = tf.reshape(tf.concat([width_detect[..., tf.newaxis], height_detect[..., tf.newaxis]], axis=-1),
                             [tf.shape(width_detect)[0], -1])
    cropped_detections = tf.concat([detections_, detections[:, 4, tf.newaxis]], axis=1)
    return cropped_image, cropped_detections


def _crop_image(image, center, size):
    cty, ctx = center
    height, width = size

    im_height, im_width = tf.cast(tf.shape(image)[0], tf.float32), tf.cast(tf.shape(image)[1], tf.float32)

    cropped_image = tf.zeros((height, width, 3), dtype=image.dtype)
    x0, x1 = tf.maximum(tf.subtract(ctx, tf.floor_div(width, 2.)), 0.), tf.minimum(tf.add(ctx, tf.floor_div(width, 2.)),
                                                                                   im_width)
    y0, y1 = tf.maximum(tf.subtract(cty, tf.floor_div(height, 2.)), 0.), tf.minimum(
        tf.add(cty, tf.floor_div(height, 2.)), im_height)

    left, right = tf.subtract(ctx, x0), tf.subtract(x1, ctx)
    top, bottom = tf.subtract(cty, y0), tf.subtract(y1, cty)

    cropped_ctx, cropped_cty = tf.floor_div(width, 2.), tf.floor_div(height, 2.)

    sliced_image = tf.slice(image, [tf.cast(y0, tf.int32), tf.cast(x0, tf.int32), 0],
                            [tf.cast(tf.subtract(y1, y0), tf.int32), tf.cast(tf.subtract(x1, x0), tf.int32), 3])
    rows = tf.range(tf.cast(tf.subtract(cropped_cty, top), tf.int32), tf.cast(tf.add(cropped_cty, bottom), tf.int32))
    cols = tf.range(tf.cast(tf.subtract(cropped_ctx, left), tf.int32), tf.cast(tf.add(cropped_ctx, right), tf.int32))
    ii, jj = tf.meshgrid(cols, rows)

    cropped_image = tf.tensor_scatter_update(cropped_image, tf.stack([jj, ii], axis=-1), sliced_image)

    border = [tf.subtract(cropped_cty, top), tf.add(cropped_cty, bottom), tf.subtract(cropped_ctx, left),
              tf.add(cropped_ctx, right)]

    offset = [tf.subtract(cty, tf.floor_div(height, 2)), tf.subtract(ctx, tf.floor_div(width, 2))]

    return cropped_image, border, offset


def full_image_crop(image, detections):
    height, width = tf.cast(tf.shape(image)[0], tf.float32), tf.cast(tf.shape(image)[1], tf.float32)

    max_hw = tf.maximum(height, width)
    center = [tf.floor_div(height, 2.), tf.floor_div(width, 2.)]
    size = [max_hw, max_hw]

    image, border, offset = _crop_image(image, center, size)

    width_detect = tf.add(detections[:, 0:4:2], border[2])
    height_detect = tf.add(detections[:, 1:4:2], border[0])
    detections_ = tf.reshape(tf.concat([width_detect[..., tf.newaxis], height_detect[..., tf.newaxis]], axis=-1),
                             [tf.shape(width_detect)[0], -1])
    detections = tf.concat([detections_, detections[:, 4, tf.newaxis]], axis=1)

    return image, detections, border


def blend_(alpha, image1, image2):
    image1 = tf.multiply(image1, alpha)
    image2 = tf.multiply(image2, tf.subtract(1., alpha))
    image1 = tf.add(image1, image2)
    return image1


def saturation_(image, gs, gs_mean, var):
    alpha = tf.add(1., tf.random.uniform([], minval=-var, maxval=var, dtype=tf.float32))
    return blend_(alpha, image, gs[:, :, None])


def brightness_(image, gs, gs_mean, var):
    alpha = tf.add(1., tf.random.uniform([], minval=-var, maxval=var, dtype=tf.float32))
    image = tf.multiply(image, alpha)
    return image


def contrast_(image, gs, gs_mean, var):
    alpha = tf.add(1., tf.random.uniform([], minval=-var, maxval=var, dtype=tf.float32))
    return blend_(alpha, image, gs_mean)


def color_jittering(image):
    ord = tf.random.shuffle(tf.range(6))[0]

    gs = tf.squeeze(tf.image.rgb_to_grayscale(tf.reverse(image, axis=[2])), axis=2)
    gs_mean = tf.reduce_mean(gs)
    val = 0.4

    def f1(img):
        img = brightness_(img, gs, gs_mean, val)
        img = contrast_(img, gs, gs_mean, val)
        img = saturation_(img, gs, gs_mean, val)
        return img

    def f2(img):
        img = brightness_(img, gs, gs_mean, val)
        img = saturation_(img, gs, gs_mean, val)
        img = contrast_(img, gs, gs_mean, val)
        return img

    def f3(img):
        img = contrast_(img, gs, gs_mean, val)
        img = brightness_(img, gs, gs_mean, val)
        img = saturation_(img, gs, gs_mean, val)
        return img

    def f4(img):
        img = saturation_(img, gs, gs_mean, val)
        img = brightness_(img, gs, gs_mean, val)
        img = contrast_(img, gs, gs_mean, val)
        return img

    def f5(img):
        img = contrast_(img, gs, gs_mean, val)
        img = saturation_(img, gs, gs_mean, val)
        img = brightness_(img, gs, gs_mean, val)
        return img

    def f6(img):
        img = saturation_(img, gs, gs_mean, val)
        img = contrast_(img, gs, gs_mean, val)
        img = brightness_(img, gs, gs_mean, val)
        return img

    image = tf.switch_case(ord, branch_fns={0: lambda: f1(image),
                                            1: lambda: f2(image),
                                            2: lambda: f3(image),
                                            3: lambda: f4(image),
                                            4: lambda: f5(image),
                                            5: lambda: f6(image)})

    return image


def lighting_(image, alphastd, eigval, eigvec):
    alpha = tf.random.normal((3,), mean=0.0, stddev=alphastd)
    image = tf.add(image, tf.tensordot(eigvec, tf.multiply(eigval, alpha), axes=1))
    return image


def print_bboxes_names(img_in, bboxes, names, thresh=0.5, name='./test.png'):
    img = np.array(img_in)
    bboxes = np.array(bboxes)
    for idx, bbox in enumerate(bboxes):
        if bbox[4] > thresh:
            color = (np.asscalar(COLORS[int(names[idx][0])][0]), np.asscalar(COLORS[int(names[idx][0])][1]),
                     np.asscalar(COLORS[int(names[idx][0])][2]))
            img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(img, names[idx][1] + str('-') + "{0:.2}".format(bbox[4]), (int(bbox[0]), int(bbox[1]) - 3),
                        cv2.FONT_HERSHEY_PLAIN, 0.8, color, 1)
    cv2.imwrite(name, img)
    return img


def print_bboxes(img, bboxes, name='./test.png'):
    img = np.array(img)
    bboxes = np.array(bboxes)
    for bbox in bboxes:
        color = (np.asscalar(COLORS[int(bbox[4])][0]), np.asscalar(COLORS[int(bbox[4])][1]),
                 np.asscalar(COLORS[int(bbox[4])][2]))
        img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
    cv2.imwrite(name, img)


def print_bbox(img, bbox, name='./test.png'):
    img = np.array(img)
    bbox = np.array(bbox)
    color = (
        np.asscalar(COLORS[int(bbox[4])][0]), np.asscalar(COLORS[int(bbox[4])][1]),
        np.asscalar(COLORS[int(bbox[4])][2]))
    img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
    cv2.imwrite(name, img)


def print_img_img(img1, img2, name='./test.png'):
    img1 = np.array(img1)
    img2 = np.array(img2)
    out = np.hstack((np.uint8(img1), np.uint8(img2)))
    cv2.imwrite(name, out)


def print_img_heatmap(img, heatmaps, name='./test'):
    img = np.array(img)
    heatmaps = np.array(heatmaps)
    for k in range(heatmaps.shape[0]):
        if heatmaps[k].sum() != 0.0:
            heatmap = cv2.applyColorMap(np.uint8(255 * heatmaps[k, :, :]), cv2.COLORMAP_JET)
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            blended_tl_img = cv2.addWeighted(heatmap, 0.9, np.uint8(img), 0.3, 0)
            cv2.imwrite(name + str(k) + '.png', blended_tl_img)


def print_gt_pred_heatmap(heatmaps, gt_heatmaps, name='./output/bbox/'):
    gt_heatmaps = np.array(gt_heatmaps)
    heatmaps = np.array(heatmaps)
    for k in range(heatmaps.shape[0]):
        if gt_heatmaps[k].sum() != 0.0:
            cv2.imwrite(name + str(k) + '.png',
                        np.hstack((np.uint8(255 * gt_heatmaps[k, :, :]), np.uint8(255 * heatmaps[k, :, :]))))


def print_img_gt_pred_heatmaps(img, heatmaps, gt_heatmaps, name='./output/bbox/'):
    img = np.array(img)
    heatmaps = np.array(heatmaps)
    gt_heatmaps = np.array(gt_heatmaps)
    for k in range(heatmaps.shape[0]):
        if gt_heatmaps[k].sum() != 0.0:
            gt_heatmap = cv2.resize(gt_heatmaps[k, :, :], (img.shape[1], img.shape[0]))
            heatmap = cv2.resize(heatmaps[k, :, :], (img.shape[1], img.shape[0]))
            heats = np.hstack(((np.uint8(255 * gt_heatmap), np.uint8(255 * heatmap))))
            cv2.imwrite(name + str(k) + '.png', np.hstack(((img[:, :, 0], heats))))
