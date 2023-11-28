from skimage.morphology import skeletonize, max_tree, binary_erosion
from scipy import ndimage, signal
import json
import glob
import scipy
import cc3d
import numpy as np
import os
import SimpleITK as sitk

def get_centerline(label, refine_times=3):
    centerline_row = centerline_extraction(label)
    centerline_row = centerline_row > 0
    centerline, endpoints, bifurcation = get_refine_skeleton(centerline_row, threshold=refine_times)
    return centerline, endpoints, bifurcation


def world_coordinate(point, spacing,):
    spacing = np.array(spacing)
    point_world = point * spacing.reshape(1, 3)
    return point_world.astype(np.float32)


def image_coordinate(point, spacing,):
    spacing = np.array(spacing)
    point_image = point / spacing.reshape(1, 3)
    point_image = np.around(point_image)
    point_image = point_image.astype(np.int16)
    return point_image


def get_refine_skeleton(skeleton, threshold=3):
    # count = 0
    # endpoints = []
    # bifurcation = []
    # while count < threshold + 1:
    #     endpoints, bifurcation = self.keypoint_detection(skeleton, mode="point")
    #     distance_map = endpoints[:, None, :] - bifurcation[None, :, :]
    #     distance_map = np.linalg.norm(distance_map, axis=-1)
    #     # get the paired endpoints and bifurcation points which are close to each other, distance < 3
    #     endpoints_index = np.min(distance_map, axis=1) < threshold
    #     endpoints_delete = endpoints[endpoints_index]
    #     count += 1
    #
    #     if not len(endpoints_delete) > 0:
    #         break
    #     elif count == threshold + 1:
    #         print("No correct yet")
    #     skeleton[endpoints_delete[:, 0], endpoints_delete[:, 1], endpoints_delete[:, 2]] = 0

    endpoints, bifurcation = keypoint_detection(skeleton, mode="point")
    distance_map = endpoints[:, None, :] - bifurcation[None, :, :]
    distance_map = np.linalg.norm(distance_map, axis=-1)
    # get the paired endpoints and bifurcation points which are close to each other, distance < 3
    endpoints_index = np.min(distance_map, axis=1) > threshold
    endpoints = endpoints[endpoints_index]

    return skeleton, endpoints, bifurcation


def sort_region(x, num=2):
    """x:3D, select six the most large region"""
    max_label = x.max()
    sum_list = [(x == index).sum() for index in range(1, int(max_label.item())+1)]
    # print(sum_list)
    # sort sum_list, return index, from large to small
    index_list = np.argsort(sum_list)[::-1]
    region_reserved = x == (index_list[0] + 1)
    for index in index_list[1:num]:
        region_reserved = region_reserved | (x == (index+1))
    return np.array(region_reserved, dtype=bool)


def fast_index(img, point):
    """point: n * 3"""
    H, W, D = img.shape
    point = point[:, 2] + D * point[:, 1] + W * D * point[:, 0]
    batch_crop = img.reshape(-1)[point]
    return batch_crop


def fast_crop(img, point):
    """
    point: n * h * w * d * 3
    return: n, h, w, d
    """
    n, h, w, d, _ = point.shape
    point = point.reshape(n * h * w * d, 3)
    batch_crop = fast_index(img, point)
    batch_crop = batch_crop.reshape(n, h, w, d)

    return batch_crop


def generate_cube(point, h, w, d):
    """
    point: n * 3
    return: n * h * w * d * 3
    """
    """"""
    n, _ = point.shape
    x = np.arange(-h, h+1)
    y = np.arange(-w, w+1)
    z = np.arange(-d, d+1)

    [Y, X, Z] = np.meshgrid(y, x, z)
    crop_region_index = np.zeros((x.shape[0] * y.shape[0] * z.shape[0], 3))
    crop_region_index[:, 0],  crop_region_index[:, 1], crop_region_index[:, 2] = \
        X.reshape(-1), Y.reshape(-1), Z.reshape(-1)
    region_index = point[:, None, :] + crop_region_index[None, :, :]
    region_index = region_index.reshape(n, x.shape[0], y.shape[0], z.shape[0], 3)
    return region_index.astype(np.int16)


def get_point_orientation(point, centerline, spacing, size=3, eps=1e-6):
    """
    :param point: 3D coordinates, havn't been multiplied by spacing
    :param centerline: 3D binary image
    :param spacing: spacing of the image, this spacing is inverse of the spacing in SimpleITK
    :param size: size of the cube, caculating the orientation in the cube
    :param eps: epsilon, because caculating the orientation is based on mean
    :return: orientation of the point
    """
    # TODO: the orientation is not correct when the discontinuity area is too small
    # padding the centerline with size
    spacing = np.array(spacing)
    centerline = np.pad(centerline, size, mode="constant", constant_values=0)
    point = point + size
    # get the cube
    index = generate_cube(point, size, size, size)
    # get the centerline in the cube
    x = index[:, :, :, :, 0]
    y = index[:, :, :, :, 1]
    z = index[:, :, :, :, 2]
    centerline_cube = centerline[x, y, z]  # n * (2 * size + 1) * (2 * size + 1) * (2 * size + 1)
    # get the centerline index
    centerline_cube[:, size, size, size] = 0  # delete the point itself
    value_exist_index = centerline_cube == 1
    cube_index = generate_cube(np.array([[size, size, size]]), size, size, size) -size  # 1 * (2 * size + 1) * (2 * size + 1) * (2 * size + 1) * 3
    centerline_coordinate = value_exist_index[:, :, :, :, None] * cube_index  # m * (2 * size + 1) * (2 * size + 1) * (2 * size + 1) * 3
    # get the orientation
    orientation = centerline_coordinate * spacing.reshape((1, 1, 1, 1, 3))  # m * (2 * size + 1) * (2 * size + 1) * (2 * size + 1) * 3
    # normalize the orientation
    orientation = orientation / (np.linalg.norm(orientation, axis=-1)[:, :, :, :, None] +
                                 eps * (1 - value_exist_index[:, :, :, :, None]))  # m * (2 * size + 1) * (2 * size + 1) * (2 * size + 1) * 3
    cube_size = (2 * size + 1)
    orientation = np.sum(orientation, axis=(1, 2, 3))   # m * 3
    # normalize the mean orientation
    orientation = orientation / np.linalg.norm(orientation, axis=-1)[:, None]  # m * 3
    # 端点相对于曲线的方向
    return -orientation


def get_point_id(point, region, mode="loop", region_threshold=0, change_region=False):
    """
    :param point: 3D coordinates
    :param region: 3D binary image
    :param mode: "loop" or "vectorize", vectorize is faster but need more memory
    :param region_threshold: threshold of region, if area of region is smaller than threshold, it will be ignored
    :param change_region: if change region, delete the region which is smaller than threshold
    :return: id of the point, also represent the order of the region in terms of area
    """
    id_map = cc3d.connected_components(region, connectivity=26)
    max_label = id_map.max()
    if mode == "loop":
        sum_list = [(id_map == index).sum() for index in range(1, int(max_label.item()) + 1)]
    elif mode == "vectorize":
        equal = id_map == np.arange(1, int(max_label.item()) + 1)[:, None, None, None]

        sum_list = np.sum(equal, axis=(1, 2, 3))
        if change_region:
            mask = equal[sum_list < region_threshold].sum(axis=0) > 0
            region[mask] = 0
    else:
        raise ValueError("mode must be loop or vectorize")

    # print(sum_list)
    # sort sum_list, return index, from large to small
    index_list = np.argsort(sum_list)[::-1]  # n * 1
    # q: what is np.argsort()
    #
    rank = np.zeros_like(sum_list)
    rank[index_list] = np.arange(len(sum_list))

    region_id = id_map[point[:, 0], point[:, 1], point[:, 2]]  # n * 1
    region_id_sort = rank[region_id - 1]  # because without background, so -1
    # region_id_sort[sum_list[region_id - 1] < region_threshold] = -1
    region_flag = (sum_list < region_threshold)[region_id - 1]
    delete_id = region_id_sort[region_flag]
    max_id = np.min(delete_id) - 1 if len(delete_id) > 0 else np.max(region_id_sort)
    if change_region:
        return region_id_sort, max_id, region
    return region_id_sort, max_id


def get_GT_box(start_points, paired_points, GT):
    """
    :param start_points: 3D coordinates, world coordinate
    :param paired_points: 3D coordinates, world coordinate
    :param GT: 3D binary image
    :return: box of the GT
    """
    id_map = cc3d.connected_components(GT, connectivity=26)
    start_points_id = id_map[start_points[:, 0], start_points[:, 1], start_points[:, 2]]
    paired_points_id = id_map[paired_points[:, 0], paired_points[:, 1], paired_points[:, 2]]
    index = (start_points_id == paired_points_id) & (start_points_id > 0)
    start_points = start_points[index]
    paired_points = paired_points[index]
    return start_points, paired_points


def get_broken_point(endpoint, region_id, orientation, threshold=3, angle_threshold_1=0.0, angle_threshold_2=0.0, paired_mode=1, region_select=None, eps=1e-6):
    """
    :param endpoint: 3D coordinates, world coordinate
    :param region_id: id of the point, belong to which region, is already sorted
    :param orientation: orientation of the point n * 3
    :param threshold: distance threshold
    :param angle_threshold_1: angle threshold between start point and link
    :param angle_threshold_2: angle threshold between link and paired point
    :param paired_mode: 0 or 1, 0: simple paired, 1: strict paired
    :param region_select: select the region which need to find paired points
    :param eps: epsilon
    """
    # sort endpoint by region_id

    region_select = region_id.max() if region_select is None else min(region_select, region_id.max())

    # get the distance map
    distance_map = endpoint[:, None, :] - endpoint[None, :, :]
    distance_map = np.linalg.norm(distance_map, axis=-1)

    # get the orientation map
    vectors = -(endpoint[:, None, :] - endpoint[None, :, :])
    vectors = vectors / (np.linalg.norm(vectors, axis=-1) + eps * (1 - (distance_map > 0)))[:, :, None]
    # the angle between link and start point is smaller than the angle between link and end point
    orientation_map_1 = np.sum(vectors * orientation[:, None, :], axis=-1)
    # the angle between link and pair point is smaller than the angle between link and start point
    orientation_map_2 = np.sum(vectors * (- orientation[None, :, :]), axis=-1)

    # set distance to 0 if the angle do not satisfy the angle_threshold
    distance_map[orientation_map_1 <= angle_threshold_1] = 0
    distance_map[orientation_map_2 <= angle_threshold_2] = 0

    # set the distance between same region to 0, the first region and the second region should not connect
    region_id_change = region_id
    region_id_change[region_id_change == 0] = 1
    select_region_index = region_id_change <= region_select
    distance_map[region_id[None, :] == region_id[:, None]] = 0

    # get the paired endpoints which are close to each other, distance < 3
    distance_map[distance_map > threshold] = 0

    # get the index of the paired endpoints
    # set 0 to inf, then get the min value
    distance_map[distance_map == 0] = np.inf
    origin_index = np.arange(len(endpoint))
    paired_index = np.argmin(distance_map, axis=1)
    # only select the origin_index which is in the region_select and the distance between them is less than inf
    inf_index = np.min(distance_map, axis=1) < np.inf
    final_index = inf_index & select_region_index
    origin_index = origin_index[final_index]
    paired_index = paired_index[final_index]
    distance_ = np.min(distance_map, axis=1)
    if paired_mode == 0:
        # get the paired endpoints
        final_paired_index = np.array([[i, j] for (i, j) in zip(origin_index, paired_index) if i < j])
    elif paired_mode == 1:
        final_paired_set = [[i, j] for (i, j) in zip(origin_index, paired_index)]
        final_paired_set = np.array(final_paired_set)
        unique_set = np.zeros(final_paired_set.shape).astype(np.int16) - 1
        count = 0
        for pairs in final_paired_set:
            if pairs[0] not in unique_set and pairs[1] not in unique_set:
                unique_set[count] = pairs
                count += 1
            else:
                if pairs[0] in unique_set:
                    index = np.where(unique_set == pairs[0])[0][0]
                else:
                    index = np.where(unique_set == pairs[1])[0][0]
                paired_exist = unique_set[index]
                distance = distance_[paired_exist[0]]
                if distance > distance_[pairs[0]]:
                    unique_set[index, 0], unique_set[index, 1] = pairs[0], pairs[1]
        final_paired_index = unique_set[:count]
    elif paired_mode == 2:
        final_paired_set = [[i, j] for (i, j) in zip(origin_index, paired_index)]
        unique_set = []
        count = 0
        for pairs in final_paired_set:
            if pairs not in unique_set and pairs[::-1] not in unique_set:
                unique_set.append(pairs)
                count += 1
        final_paired_index = np.array(unique_set)
    else:
        raise ValueError("paired_mode should be 0, 1, 2")


    # get the paired endpoints
    if len(final_paired_index) == 0:
        return None, None, None, None
    start_point = endpoint[final_paired_index[:, 0]]
    paired_endpoint = endpoint[final_paired_index[:, 1]]
    start_point_orientation = orientation[final_paired_index[:, 0]]
    paired_endpoint_orientation = orientation[final_paired_index[:, 1]]
    return start_point, paired_endpoint, start_point_orientation, paired_endpoint_orientation


def get_broken_point_with_conditions(endpoint, region_id, condition_id, orientation, threshold=3, angle_threshold_1=0.0,
                                     angle_threshold_2=0.0, paired_mode=1, region_select=None, eps=1e-6):
    """
    :param endpoint: 3D coordinates, world coordinate
    :param region_id: id of the point, belong to which region, is already sorted
    :param condition_id: condition id of the point, like constriction on last fracture region, ground truth ...
    ...and auxiliary segmentation, n * m, m is the number of conditions, if the point belong to same condition_id, or...
    ...the condition_id is 0, then the point can not be paired
    :param orientation: orientation of the point n * 3
    :param threshold: distance threshold
    :param angle_threshold_1: angle threshold between start point and link
    :param angle_threshold_2: angle threshold between link and paired point
    :param paired_mode: 0 or 1, 0: simple paired, 1: strict paired
    :param region_select: select the region which need to find paired points
    :param eps: epsilon
    """
    # sort endpoint by region_id
    if len(region_id) == 0:
        return None, None, None, None
    region_select = region_id.max() if region_select is None else min(region_select, region_id.max())

    # get the distance map
    distance_map = endpoint[:, None, :] - endpoint[None, :, :]
    distance_map = np.linalg.norm(distance_map, axis=-1)

    # get the orientation map
    vectors = -(endpoint[:, None, :] - endpoint[None, :, :])
    vectors = vectors / (np.linalg.norm(vectors, axis=-1) + eps * (1 - (distance_map > 0)))[:, :, None]
    # the angle between link and start point is smaller than the angle between link and end point
    orientation_map_1 = np.sum(vectors * orientation[:, None, :], axis=-1)
    # the angle between link and pair point is smaller than the angle between link and start point
    orientation_map_2 = np.sum(vectors * (- orientation[None, :, :]), axis=-1)

    # set distance to 0 if the angle do not satisfy the angle_threshold
    distance_map[orientation_map_1 <= angle_threshold_1] = 0
    distance_map[orientation_map_2 <= angle_threshold_2] = 0

    # set the distance between same region to 0, the first region and the second region should not connect
    region_id_change = region_id
    region_id_change[region_id_change == 0] = 1
    select_region_index = region_id_change <= region_select
    distance_map[region_id[None, :] == region_id[:, None]] = 0

    # set the distance between to 0 using the condition_id,
    # distance_map[label_gt_id[None, :] != label_gt_id[:, None]] = 0
    if condition_id is not None:
        condition_number = condition_id.shape[1]
        for i in range(condition_number):
            distance_map[condition_id[:, i][None, :] != condition_id[:, i][:, None]] = 0
            distance_map[condition_id[:, i] == 0, :] = 0
            distance_map[:, condition_id[:, i] == 0] = 0

    # get the paired endpoints which are close to each other, distance < 3
    distance_map[distance_map > threshold] = 0

    # get the index of the paired endpoints
    # set 0 to inf, then get the min value
    distance_map[distance_map == 0] = np.inf
    origin_index = np.arange(len(endpoint))
    paired_index = np.argmin(distance_map, axis=1)
    # only select the origin_index which is in the region_select and the distance between them is less than inf
    inf_index = np.min(distance_map, axis=1) < np.inf
    final_index = inf_index & select_region_index
    origin_index = origin_index[final_index]
    paired_index = paired_index[final_index]
    distance_ = np.min(distance_map, axis=1)
    if paired_mode == 0:
        # get the paired endpoints
        final_paired_index = np.array([[i, j] for (i, j) in zip(origin_index, paired_index) if i < j])
    elif paired_mode == 1:
        final_paired_set = [[i, j] for (i, j) in zip(origin_index, paired_index)]
        final_paired_set = np.array(final_paired_set)
        unique_set = np.zeros(final_paired_set.shape).astype(np.int16) - 1
        count = 0
        for pairs in final_paired_set:
            if pairs[0] not in unique_set and pairs[1] not in unique_set:
                unique_set[count] = pairs
                count += 1
            else:
                if pairs[0] in unique_set:
                    index = np.where(unique_set == pairs[0])[0][0]
                else:
                    index = np.where(unique_set == pairs[1])[0][0]
                paired_exist = unique_set[index]
                distance = distance_[paired_exist[0]]
                if distance > distance_[pairs[0]]:
                    unique_set[index, 0], unique_set[index, 1] = pairs[0], pairs[1]
        final_paired_index = unique_set[:count]
    elif paired_mode == 2:
        final_paired_set = [[i, j] for (i, j) in zip(origin_index, paired_index)]
        unique_set = []
        count = 0
        for pairs in final_paired_set:
            if pairs not in unique_set and pairs[::-1] not in unique_set:
                unique_set.append(pairs)
                count += 1

        final_paired_index = np.array(unique_set)
    else:
        raise ValueError("paired_mode should be 0, 1, 2")

    # get the paired endpoints
    if len(final_paired_index) == 0:
        return None, None, None, None
    start_point = endpoint[final_paired_index[:, 0]]
    paired_endpoint = endpoint[final_paired_index[:, 1]]
    start_orientation = orientation[final_paired_index[:, 0]]
    paired_orientation = orientation[final_paired_index[:, 1]]
    return start_point, paired_endpoint, start_orientation, paired_orientation


def delete_point_by_orientation(start_point, paired_endpoint, start_point_orientation, threshold: float = 0.0):
    """
    :param start_point: 3D coordinates
    :param paired_endpoint: 3D coordinates
    :param start_point_orientation: orientation of the start point
    :param threshold: threshold of the angle, cos(angle), 0 stands for 90 degree
    """
    # get the vector of the paired endpoints
    vector = (paired_endpoint - start_point)
    vector = vector / np.linalg.norm(vector, axis=-1)[:, None]
    # get the angle between the vector and the orientation
    angle = np.sum(vector * start_point_orientation, axis=-1)
    # delete the paired endpoints which the angle is larger than threshold
    remain_index = angle > threshold

    return remain_index


def get_broken_cube(start_point, end_point, shape, cube_min_size=(10, 10, 10)):
    """
    :param start_point: 3D coordinates, such paired endpoint determine the rectangle, shape (n, 3)
    :param end_point: 3D coordinates
    :param shape: shape of the image
    :param cube_min_size: the min size of the cube
    """
    # get the rectangle
    mask = np.zeros(shape, dtype=bool)
    x = np.min(np.concatenate((start_point[:, :1], end_point[:, :1]), axis=1), axis=1)
    x1 = np.max(np.concatenate((start_point[:, :1], end_point[:, :1]), axis=1), axis=1)
    y = np.min(np.concatenate((start_point[:, 1:2], end_point[:, 1:2]), axis=1), axis=1)
    y1 = np.max(np.concatenate((start_point[:, 1:2], end_point[:, 1:2]), axis=1), axis=1)
    z = np.min(np.concatenate((start_point[:, 2:], end_point[:, 2:]), axis=1), axis=1)
    z1 = np.max(np.concatenate((start_point[:, 2:], end_point[:, 2:]), axis=1), axis=1)
    x_refine = x1 - x - cube_min_size[0]
    y_refine = y1 - y - cube_min_size[1]
    z_refine = z1 - z - cube_min_size[2]
    x_index = x_refine < 0
    y_index = y_refine < 0
    z_index = z_refine < 0
    x[x_index] = np.maximum(x[x_index] - cube_min_size[0] // 2, 0)
    x1[x_index] = np.minimum(x1[x_index] + cube_min_size[0] // 2, shape[0] - 1)
    y[y_index] = np.maximum(y[y_index] - cube_min_size[1] // 2, 0)
    y1[y_index] = np.minimum(y1[y_index] + cube_min_size[1] // 2, shape[1] - 1)
    z[z_index] = np.maximum(z[z_index] - cube_min_size[2] // 2, 0)
    z1[z_index] = np.minimum(z1[z_index] + cube_min_size[2] // 2, shape[2] - 1)

    for i in range(len(x)):
        mask[x[i]:x1[i] + 1, y[i]:y1[i] + 1, z[i]:z1[i] + 1] = True
    return mask, len(x)


def get_broken_sphere(start_point, end_point, shape, spacing, dilation, min_radius, mode='loop'):
    """
    :param start_point: 3D coordinates, such paired endpoint determine the rectangle, shape (n, 3)
    :param end_point: 3D coordinates
    :param shape: shape of the image
    :param spacing: spacing of the image
    :param dilation: dilation of the sphere, percentage of the radius
    :param min_radius: min radius of the sphere, world coordinate
    :param mode: 'loop' or 'vector'
    """

    mask = np.zeros(shape, dtype=bool)
    spacing = np.array(spacing).reshape(1, 3)
    start_point_float = start_point.astype(np.float32)
    end_point_float = end_point.astype(np.float32)
    x_mean = (start_point_float[:, 0] + end_point_float[:, 0]) / 2
    y_mean = (start_point_float[:, 1] + end_point_float[:, 1]) / 2
    z_mean = (start_point_float[:, 2] + end_point_float[:, 2]) / 2
    radius_world = np.sqrt(np.sum((start_point_float * spacing - end_point_float * spacing) ** 2, axis=-1)) / 2
    radius_world[radius_world < min_radius] = min_radius
    # generate the meshgrid
    x = np.arange(shape[0])
    y = np.arange(shape[1])
    z = np.arange(shape[2])
    x, y, z = np.meshgrid(x, y, z, indexing='ij')
    x_float = x.astype(np.float32)
    y_float = y.astype(np.float32)
    z_float = z.astype(np.float32)
    if mode == 'loop':
        for i in range(len(start_point)):
            distance_world = np.sqrt(((x_float - x_mean[i, None, None, None]) * spacing[0, 0]) ** 2 +
                                     ((y_float - y_mean[i, None, None, None]) * spacing[0, 1]) ** 2 +
                                     ((z_float - z_mean[i, None, None, None]) * spacing[0, 2]) ** 2)
            index = (radius_world[i, None, None, None] * (1 + dilation)) > distance_world
            mask[x[index], y[index], z[index]] = True
    elif mode == 'vector':
        distance_world = np.sqrt(((x_float[None, ...] - x_mean[:, None, None, None]) * spacing[0, 0]) ** 2 +
                                 ((y_float[None, ...] - y_mean[:, None, None, None]) * spacing[0, 1]) ** 2 +
                                 ((z_float[None, ...] - z_mean[:, None, None, None]) * spacing[0, 2]) ** 2)
        mask[distance_world < (radius_world[:, None, None, None] * (1 + dilation))] = True
        mask = np.sum(mask, axis=0) > 0
    return mask, len(start_point)


def save_paired_point_txt(start_point, paired_point, path):
    """
    :param start_point: 3D coordinates, such paired endpoint determine the rectangle, shape (n, 3)
    :param end_point: 3D coordinates
    :param path: path to save the txt file
    """
    # use json to save the data
    data = {'start_point': start_point.tolist(), 'paired_point': paired_point.tolist()}
    with open(path, 'w') as outfile:
        json.dump(data, outfile)


def centerline_extraction(label):
    """
    :param label: 3D binary image
    :return: 3D binary image, centerline
    """
    # skeletonize
    skeleton = skeletonize(label)
    return skeleton


def keypoint_detection(img, mode="image"):
    """
    :param img: 3D binary image, centerline
    :param mode: "image" or "point"
    :return: 3D binary image if mode is image, else return the coordinates of endpoints and bifurcation
    """
    # Kernel to sum the neighbours
    kernel_3d = np.ones((3, 3, 3))
    kernel_3d[1, 1, 1] = 0

    # 3D convolution (cast image to int32 to avoid overflow)
    img_conv = signal.convolve(img.astype(np.int32), kernel_3d, mode='same')
    img_around = np.around(img_conv).astype(np.uint16)
    endpoints = np.stack(np.where((img == 1) & (img_around == 1)), axis=1)
    # ### 边界的值不能过大，应该在一个范围内可能比较合适：3<sum<26？上限26,需要验证
    bifurcation = np.stack(np.where((img == 1) & ((img_around <= 26) & (img_around >= 3))), axis=1)
    if mode == "image":
        point_map = np.zeros_like(img)
        point_map_bifurcation = np.zeros_like(img)
        point_map[endpoints[:, 0], endpoints[:, 1], endpoints[:, 2]] = 1

        point_map_bifurcation[bifurcation[:, 0], bifurcation[:, 1], bifurcation[:, 2]] = 1

        return point_map.astype(np.uint16), point_map_bifurcation.astype(np.uint16)
    elif mode == "point":
        return endpoints, bifurcation


def save(x, spacing, origin, direction, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    x = sitk.GetImageFromArray(x)
    x.SetOrigin(origin)
    x.SetSpacing(spacing)
    x.SetDirection(direction)
    sitk.WriteImage(x, save_path)


def dilation(x, times=1):
    x = np.array(x.squeeze())
    struct1 = ndimage.generate_binary_structure(3, 1)
    x = ndimage.binary_dilation(x, structure=struct1, iterations=times).astype(x.dtype)
    return x


def second_stage_merge(m1, m2, mask):
    merge = m1 * (1 - mask) + mask * (m2 > 0)
    merge_mask = mask * (m2 > 0)
    return merge, merge_mask


def select_two_biggest_connected_region(region):
    region_mask = cc3d.connected_components(region > 0, connectivity=6)
    region_two = region * sort_region(region_mask, num=2)
    return region_two


def sparrowlink_metric(label,
                       GT,
                       view=False,
                       spacing=None,
                       origin=None,
                       direction=None,
                       max_broken_size_percentage=0.1,
                       min_sphere_radius_percentage=0.02,
                       sphere_dilation_percentage=0.1,
                       save_path=None,
                       cube_min_size=(10, 10, 10),
                       skeleton_refine_times=3,
                       region_threshold=5,
                       angle_threshold_1=0.0,
                       angle_threshold_2=0.0, ):
    """
    :param label: the label of the image or segmentation
    :param GT: the ground truth of the segmentation
    :param view: whether to save the image
    :param spacing: the spacing of the image
    :param origin: the origin of the image
    :param direction: the direction of the image
    :param max_broken_size_percentage: the max broken size percentage, used in constriction on distance
    :param min_sphere_radius_percentage: the min sphere radius percentage, used for some small fracture
    :param sphere_dilation_percentage: the sphere dilation percentage, to consider the structure around fracture area
    :param save_path: the path to save the discontinuity detection results
    :param region_select: the region select, matching points consist of start point and paired point, start point is...
    ...constricted in the region with region_select.
    :param cube_min_size: the min size of the cube, used for small fracture
    :param skeleton_refine_times: the refine times of the skeleton, zhang algorithm might cause some small branch in the...
    ...skeleton, so we need to refine the skeleton to delete the small branch
    :param region_threshold: the region threshold, used to delete the small region in the segmentation
    :param angle_threshold_1: the angle threshold 1, used for constriction on orientation
    :param angle_threshold_2: the angle threshold 2, used for constriction on orientation
    """
    save_path = str(save_path) if save_path is not None else None
    patient_name = save_path[-16: -7] if save_path is not None else ''
    shape = label.shape
    max_broken_size = np.sqrt((shape[0] * spacing[2]) ** 2 +
                              (shape[1] * spacing[1]) ** 2 +
                              (shape[2] * spacing[0]) ** 2) * max_broken_size_percentage
    min_sphere_radius = np.sqrt((shape[0] * spacing[2]) ** 2 +
                                (shape[1] * spacing[1]) ** 2 +
                                (shape[2] * spacing[0]) ** 2) * min_sphere_radius_percentage

    # get centerline and row endpoints
    centerline, endpoints, bifurcation = get_centerline(label, refine_times=skeleton_refine_times)

    # get the region id(sort by area) of endpoints using cc3d, and delete the region which is too small
    region_id, max_id, centerline = get_point_id(endpoints, centerline, mode="vectorize",
                                                 region_threshold=region_threshold, change_region=True)
    delete_id = max_id
    index = np.argsort(region_id)
    endpoints = endpoints[index]
    region_id = region_id[index]

    remain_index = region_id <= delete_id
    endpoints = endpoints[remain_index]
    region_id = region_id[remain_index]

    # get the world coordinate of endpoints
    endpoints_world = world_coordinate(endpoints, spacing[::-1])

    # get the orientation of endpoints
    endpoints_orientation_world = get_point_orientation(endpoints, centerline, spacing[::-1], size=3, )

    # for later GT detection
    endpoints_for_gt = endpoints.copy()
    endpoints_orientation_world_for_gt = endpoints_orientation_world.copy()
    # get the label_gt_id
    # 联影老六给的标签有错误，只取两个最大连通域

    GT_refine_mask = sort_region(cc3d.connected_components(GT > 0, connectivity=26), num=2)
    GT = GT * GT_refine_mask

    label_gt_id = GT[endpoints_for_gt[:, 0], endpoints_for_gt[:, 1], endpoints_for_gt[:, 2]]
    index = label_gt_id > 0

    # delete background endpoints
    endpoints_for_gt = endpoints_for_gt[index]
    region_id = region_id[index]
    endpoints_orientation_world_for_gt = endpoints_orientation_world_for_gt[index]
    label_gt_id = label_gt_id[index][:, None]
    endpoints_world_for_gt = world_coordinate(endpoints_for_gt, spacing[::-1])
    if endpoints_world is not None:
        start_point_gt_world, paired_point_gt_world, _, _ = \
            get_broken_point_with_conditions(endpoints_world_for_gt, region_id,
                                             condition_id=label_gt_id,
                                             orientation=endpoints_orientation_world_for_gt,
                                             threshold=max_broken_size,
                                             angle_threshold_1=angle_threshold_1,
                                             angle_threshold_2=-1,
                                             paired_mode=2,
                                             region_select=None, )
    else:
        start_point_gt_world, paired_point_gt_world = None, None

    if start_point_gt_world is not None:
        start_point_gt = image_coordinate(start_point_gt_world, spacing[::-1])
        paired_point_gt = image_coordinate(paired_point_gt_world, spacing[::-1])
        mask_gt, num_gt = get_broken_cube(start_point=start_point_gt, end_point=paired_point_gt, shape=label.shape,
                                          cube_min_size=cube_min_size)
        mask_sphere_gt, num_sphere_gt = get_broken_sphere(start_point=start_point_gt, end_point=paired_point_gt,
                                                          spacing=spacing[::-1], shape=label.shape,
                                                          dilation=sphere_dilation_percentage,
                                                          min_radius=min_sphere_radius, mode="loop")
    else:
        start_point_gt, paired_point_gt = None, None
        mask_gt, num_gt = np.zeros(shape=label.shape, dtype=np.uint16), 0
        mask_sphere_gt, num_sphere_gt = np.zeros(shape=label.shape, dtype=np.uint16), 0

    if view and save_path is not None:
        save(x=mask_gt.astype(np.uint16), save_path=save_path.replace(".nii.gz", "_cube_GT.nii.gz"),
             spacing=spacing,
             origin=origin, direction=direction)
        save(x=mask_sphere_gt.astype(np.uint16), save_path=save_path.replace(".nii.gz", "_sphere_GT.nii.gz"),
             spacing=spacing, origin=origin, direction=direction)
        if start_point_gt is not None:
            discontinuity_detection = {"num": start_point_gt.shape[0], "start_point": start_point_gt.tolist(),
                                       "paired_point": paired_point_gt.tolist(), "spacing": spacing[::-1], }
            with open(save_path.replace(".nii.gz", "_gt.json"), 'w') as f:
                json.dump(discontinuity_detection, f, indent=1)
        else:
            discontinuity_detection = {"num": 0, "start_point": np.array([[]]).tolist(),
                                       "paired_point": np.array([[]]).tolist(), "spacing": spacing[::-1], }
            with open(save_path.replace(".nii.gz", "_gt.json"), 'w') as f:
                json.dump(discontinuity_detection, f, indent=1)

    result_dict={"mask_gt": mask_gt, "mask_sphere_gt": mask_sphere_gt, "num_gt": num_gt, "name": patient_name}

    return result_dict

def update(pbar, L, L_gt, L_auxiliary, result):
    pbar.update()
    if dict.get(result, "num", None) is not None:
        L.append({"num": result["num"], "name": result["name"]})
    if dict.get(result, "num_gt", None) is not None:
        L_gt.append({"num": result["num_gt"], "name": result["name"]})
    if dict.get(result, "num_auxiliary", None) is not None:
        L_auxiliary.append({"num": result["num_auxiliary"], "name": result["name"]})


def error_back(err):
    print(err)