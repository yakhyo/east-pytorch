import math
import os

import cv2
import numpy as np
import torch
from PIL import Image
from shapely.geometry import Polygon


def distance(x1, y1, x2, y2):
    """Euclidean Distance"""
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def move_points(vertices, index1, index2, r, coef):
    """Move the two points to shrink edge
    Args:
        vertices: vertices of text region <numpy.ndarray, (8,)>
        index1  : offset of point1
        index2  : offset of point2
        r       : [r1, r2, r3, r4] in paper
        coef    : shrink ratio in paper
    Return:
        vertices: vertices where one edge has been shinked
    """
    index1 = index1 % 4
    index2 = index2 % 4
    x1_index = index1 * 2 + 0
    y1_index = index1 * 2 + 1
    x2_index = index2 * 2 + 0
    y2_index = index2 * 2 + 1

    r1 = r[index1]
    r2 = r[index2]
    length_x = vertices[x1_index] - vertices[x2_index]
    length_y = vertices[y1_index] - vertices[y2_index]
    length = distance(vertices[x1_index], vertices[y1_index], vertices[x2_index], vertices[y2_index])
    if length > 1:
        ratio = (r1 * coef) / length
        vertices[x1_index] += ratio * (-length_x)
        vertices[y1_index] += ratio * (-length_y)
        ratio = (r2 * coef) / length
        vertices[x2_index] += ratio * length_x
        vertices[y2_index] += ratio * length_y
    return vertices


def shrink_poly(vertices, coef=0.3):
    """Shrink the text region
    Args:
        vertices: vertices of text region <numpy.ndarray, (8,)>
        coef    : shrink ratio in paper
    Return:
        v       : vertices of shrinked text region <numpy.ndarray, (8,)>
    """
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    r1 = min(distance(x1, y1, x2, y2), distance(x1, y1, x4, y4))
    r2 = min(distance(x2, y2, x1, y1), distance(x2, y2, x3, y3))
    r3 = min(distance(x3, y3, x2, y2), distance(x3, y3, x4, y4))
    r4 = min(distance(x4, y4, x1, y1), distance(x4, y4, x3, y3))
    r = [r1, r2, r3, r4]

    # obtain offset to perform move_points() automatically
    if distance(x1, y1, x2, y2) + distance(x3, y3, x4, y4) > distance(x2, y2, x3, y3) + distance(x1, y1, x4, y4):
        offset = 0  # two longer edges are (x1y1-x2y2) & (x3y3-x4y4)
    else:
        offset = 1  # two longer edges are (x2y2-x3y3) & (x4y4-x1y1)

    v = vertices.copy()
    v = move_points(v, 0 + offset, 1 + offset, r, coef)
    v = move_points(v, 2 + offset, 3 + offset, r, coef)
    v = move_points(v, 1 + offset, 2 + offset, r, coef)
    v = move_points(v, 3 + offset, 4 + offset, r, coef)
    return v


def get_rotate_mat(theta):
    """positive theta value means rotate clockwise"""
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])


def rotate_vertices(vertices, theta, anchor=None):
    """Rotate vertices around anchor
    Args:
        vertices: vertices of text region <numpy.ndarray, (8,)>
        theta   : angle in radian measure
        anchor  : fixed position during rotation
    Return:
        rotated vertices <numpy.ndarray, (8,)>
    """
    v = vertices.reshape((4, 2)).T
    if anchor is None:
        anchor = v[:, :1]
    rotate_mat = get_rotate_mat(theta)
    res = np.dot(rotate_mat, v - anchor)
    return (res + anchor).T.reshape(-1)


def get_boundary(vertices):
    """Get the tight boundary around given vertices
    Args:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Return:
        the boundary
    """
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    x_min = min(x1, x2, x3, x4)
    x_max = max(x1, x2, x3, x4)
    y_min = min(y1, y2, y3, y4)
    y_max = max(y1, y2, y3, y4)
    return x_min, x_max, y_min, y_max


def calculate_error(vertices):
    """Default orientation is x1y1 : left-top, x2y2 : right-top, x3y3 : right-bot, x4y4 : left-bot
    calculate the difference between the vertices orientation and default orientation
    Args:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Return:
        err     : difference measure
    """
    x_min, x_max, y_min, y_max = get_boundary(vertices)
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    err = (
            distance(x1, y1, x_min, y_min)
            + distance(x2, y2, x_max, y_min)
            + distance(x3, y3, x_max, y_max)
            + distance(x4, y4, x_min, y_max)
    )
    return err


def find_min_rect_angle(vertices):
    """find the best angle to rotate poly and obtain min rectangle
    Args:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Return:
        the best angle <radian measure>
    """
    angle_interval = 1
    angle_list = list(range(-90, 90, angle_interval))
    area_list = []
    for theta in angle_list:
        rotated = rotate_vertices(vertices, theta / 180 * math.pi)
        x1, y1, x2, y2, x3, y3, x4, y4 = rotated
        temp_area = (max(x1, x2, x3, x4) - min(x1, x2, x3, x4)) * (max(y1, y2, y3, y4) - min(y1, y2, y3, y4))
        area_list.append(temp_area)

    sorted_area_index = sorted(list(range(len(area_list))), key=lambda k: area_list[k])
    min_error = float("inf")
    best_index = -1
    rank_num = 10
    # find the best angle with correct orientation
    for index in sorted_area_index[:rank_num]:
        rotated = rotate_vertices(vertices, angle_list[index] / 180 * math.pi)
        temp_error = calculate_error(rotated)
        if temp_error < min_error:
            min_error = temp_error
            best_index = index
    return angle_list[best_index] / 180 * math.pi


def is_cross_text(start_loc, length, vertices):
    """check if the crop image crosses text regions
    Args:
        start_loc: left-top position
        length   : length of crop image
        vertices : vertices of text regions <numpy.ndarray, (n,8)>
    Return:
        True if crop image crosses text region
    """
    if vertices.size == 0:
        return False
    start_w, start_h = start_loc
    a = np.array(
        [start_w, start_h, start_w + length, start_h, start_w + length, start_h + length, start_w, start_h + length]
    ).reshape((4, 2))
    p1 = Polygon(a).convex_hull
    for vertice in vertices:
        p2 = Polygon(vertice.reshape((4, 2))).convex_hull
        inter = p1.intersection(p2).area
        if 0.01 <= inter / p2.area <= 0.99:
            return True
    return False


def crop(image, vertices, labels, length):
    """crop image patches to obtain batch and augment
    Args:
        image         : PIL Image
        vertices    : vertices of text regions <numpy.ndarray, (n,8)>
        labels      : 1->valid, 0->ignore, <numpy.ndarray, (n,)>
        length      : length of cropped image region
    Return:
        region      : cropped image region
        new_vertices: new vertices in cropped region
    """
    h, w = image.height, image.width
    # confirm the shortest side of image >= length
    if h >= w and w < length:
        image = image.resize((length, int(h * length / w)), Image.BILINEAR)
    elif h < w and h < length:
        image = image.resize((int(w * length / h), length), Image.BILINEAR)
    ratio_w = image.width / w
    ratio_h = image.height / h
    assert ratio_w >= 1 and ratio_h >= 1

    new_vertices = np.zeros(vertices.shape)
    if vertices.size > 0:
        new_vertices[:, [0, 2, 4, 6]] = vertices[:, [0, 2, 4, 6]] * ratio_w
        new_vertices[:, [1, 3, 5, 7]] = vertices[:, [1, 3, 5, 7]] * ratio_h

    # find random position
    remain_h = image.height - length
    remain_w = image.width - length
    flag = True
    cnt = 0
    while flag and cnt < 1000:
        cnt += 1
        start_w = int(np.random.rand() * remain_w)
        start_h = int(np.random.rand() * remain_h)
        flag = is_cross_text([start_w, start_h], length, new_vertices[labels == 1, :])
    box = (start_w, start_h, start_w + length, start_h + length)
    region = image.crop(box)
    if new_vertices.size == 0:
        return region, new_vertices

    new_vertices[:, [0, 2, 4, 6]] -= start_w
    new_vertices[:, [1, 3, 5, 7]] -= start_h
    return region, new_vertices


def rotate_all_pixels(rotate_mat, anchor_x, anchor_y, length):
    """Get rotated locations of all pixels for next stages
    Args:
        rotate_mat: rotatation matrix
        anchor_x  : fixed x position
        anchor_y  : fixed y position
        length    : length of image
    Return:
        rotated_x : rotated x positions <numpy.ndarray, (length,length)>
        rotated_y : rotated y positions <numpy.ndarray, (length,length)>
    """
    x = np.arange(length)
    y = np.arange(length)
    x, y = np.meshgrid(x, y)
    x_lin = x.reshape((1, x.size))
    y_lin = y.reshape((1, x.size))
    coord_mat = np.concatenate((x_lin, y_lin), 0)
    rotated_coord = np.dot(rotate_mat, coord_mat - np.array([[anchor_x], [anchor_y]])) + np.array(
        [[anchor_x], [anchor_y]]
    )
    rotated_x = rotated_coord[0, :].reshape(x.shape)
    rotated_y = rotated_coord[1, :].reshape(y.shape)
    return rotated_x, rotated_y


def adjust_height(image, vertices, ratio=0.2):
    """Adjust height of image to aug data
    Args:
        image         : PIL Image
        vertices    : vertices of text regions <numpy.ndarray, (n,8)>
        ratio       : height changes in [0.8, 1.2]
    Return:
        image         : adjusted PIL Image
        new_vertices: adjusted vertices
    """
    ratio_h = 1 + ratio * (np.random.rand() * 2 - 1)
    old_h = image.height
    new_h = int(np.around(old_h * ratio_h))
    image = image.resize((image.width, new_h), Image.BILINEAR)

    new_vertices = vertices.copy()
    if vertices.size > 0:
        new_vertices[:, [1, 3, 5, 7]] = vertices[:, [1, 3, 5, 7]] * (new_h / old_h)
    return image, new_vertices


def rotate(image, vertices, angle_range=10):
    """Rotate image [-10, 10] degree to aug data
    Args:
        image         : PIL Image
        vertices    : vertices of text regions <numpy.ndarray, (n,8)>
        angle_range : rotate range
    Return:
        image         : rotated PIL Image
        new_vertices: rotated vertices
    """
    center_x = (image.width - 1) / 2
    center_y = (image.height - 1) / 2
    angle = angle_range * (np.random.rand() * 2 - 1)
    image = image.rotate(angle, Image.BILINEAR)
    new_vertices = np.zeros(vertices.shape)
    for i, vertice in enumerate(vertices):
        new_vertices[i, :] = rotate_vertices(vertice, -angle / 180 * math.pi, np.array([[center_x], [center_y]]))
    return image, new_vertices


def get_score_geo(image, vertices, labels, scale, length):
    """Generate score gt and geometry gt
    Args:
        image     : PIL Image
        vertices: vertices of text regions <numpy.ndarray, (n,8)>
        labels  : 1->valid, 0->ignore, <numpy.ndarray, (n,)>
        scale   : feature map / image
        length  : image length
    Return:
        score gt, geo gt, ignored
    """
    score_map = np.zeros((int(image.height * scale), int(image.width * scale), 1), np.float32)
    geo_map = np.zeros((int(image.height * scale), int(image.width * scale), 5), np.float32)
    ignored_map = np.zeros((int(image.height * scale), int(image.width * scale), 1), np.float32)

    index = np.arange(0, length, int(1 / scale))
    index_x, index_y = np.meshgrid(index, index)
    ignored_polys = []
    polys = []

    for i, vertice in enumerate(vertices):
        if labels[i] == 0:
            ignored_polys.append(np.around(scale * vertice.reshape((4, 2))).astype(np.int32))
            continue

        poly = np.around(scale * shrink_poly(vertice).reshape((4, 2))).astype(np.int32)  # scaled & shrinked
        polys.append(poly)
        temp_mask = np.zeros(score_map.shape[:-1], np.float32)
        cv2.fillPoly(temp_mask, [poly], 1)

        theta = find_min_rect_angle(vertice)
        rotate_mat = get_rotate_mat(theta)

        rotated_vertices = rotate_vertices(vertice, theta)
        x_min, x_max, y_min, y_max = get_boundary(rotated_vertices)
        rotated_x, rotated_y = rotate_all_pixels(rotate_mat, vertice[0], vertice[1], length)

        d1 = rotated_y - y_min
        d1[d1 < 0] = 0
        d2 = y_max - rotated_y
        d2[d2 < 0] = 0
        d3 = rotated_x - x_min
        d3[d3 < 0] = 0
        d4 = x_max - rotated_x
        d4[d4 < 0] = 0
        geo_map[:, :, 0] += d1[index_y, index_x] * temp_mask
        geo_map[:, :, 1] += d2[index_y, index_x] * temp_mask
        geo_map[:, :, 2] += d3[index_y, index_x] * temp_mask
        geo_map[:, :, 3] += d4[index_y, index_x] * temp_mask
        geo_map[:, :, 4] += theta * temp_mask

    cv2.fillPoly(ignored_map, ignored_polys, 1)
    cv2.fillPoly(score_map, polys, 1)

    score_map = torch.Tensor(score_map).permute(2, 0, 1)
    geo_map = torch.Tensor(geo_map).permute(2, 0, 1)
    ignored_map = torch.Tensor(ignored_map).permute(2, 0, 1)

    return score_map, geo_map, ignored_map


def extract_vertices(lines):
    """Extract vertices info from txt lines
    Args:
        lines   : list of string info
    Return:
        vertices: vertices of text regions <numpy.ndarray, (n,8)>
        labels  : 1->valid, 0->ignore, <numpy.ndarray, (n,)>
    """
    labels = []
    vertices = []
    for line in lines:
        label = 0 if "###" in line else 1
        coord = list(map(int, line.rstrip("\n").lstrip("\ufeff").split(",")[:8]))

        vertices.append(coord)
        labels.append(label)

    return np.array(vertices), np.array(labels)


def strip_optimizer(s, f="model_f16.pt"):
    x = torch.load(s, map_location=torch.device("cpu"))
    for k in "optimizer", "updates", "best_fitness":  # keys
        x[k] = None
    x["epoch"] = -1  # ignore for now
    x["model"].half()  # to FP16
    for p in x["model"].parameters():
        p.requires_grad = False
    torch.save(x, s.replace("model.ckpt", "model_f16.pt"))
    file_size = os.path.getsize(f) / 1e6
    print(f"Optimizer stripped from {s},{(' saved as %s,' % f)} {file_size:.1f}MB")
