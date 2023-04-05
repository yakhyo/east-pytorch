import argparse
import os

import lanms
import numpy as np
import torch

from east.models import EAST
from east.utils import get_rotate_mat
from PIL import Image, ImageDraw
from torchvision import transforms


def resize(image):
    """Resize image to be divisible by 32"""
    old_w, old_h = image.size
    # new height and width
    new_h = old_h if old_h % 32 == 0 else (old_h // 32) * 32
    new_w = old_w if old_w % 32 == 0 else (old_w // 32) * 32
    # resize to new height and width
    image = image.resize((new_w, new_h), Image.BILINEAR)

    ratio_h = new_h / old_h
    ratio_w = new_w / old_w

    return image, ratio_h, ratio_w


def is_valid_poly(res, score_shape, scale):
    """Check if the poly in image scope
    Args:
        res        : restored poly in original image
        score_shape: score map shape
        scale      : feature map -> image
    Return:
        True if valid
    """
    cnt = 0
    for i in range(res.shape[1]):
        if res[0, i] < 0 or res[0, i] >= score_shape[1] * scale or res[1, i] < 0 or res[1, i] >= score_shape[0] * scale:
            cnt += 1
    return True if cnt <= 1 else False


def restore_polys(valid_pos, valid_geo, score_shape, scale=4):
    """Restore polys from feature maps in given positions
    Args:
        valid_pos  : potential text positions <numpy.ndarray, (n,2)>
        valid_geo  : geometry in valid_pos <numpy.ndarray, (5,n)>
        score_shape: shape of score map
        scale      : image / feature map
    Return:
        restored polys <numpy.ndarray, (n,8)>, index
    """
    polys = []
    index = []
    valid_pos *= scale
    d = valid_geo[:4, :]  # 4 x N
    angle = valid_geo[4, :]  # N,

    for i in range(valid_pos.shape[0]):
        x = valid_pos[i, 0]
        y = valid_pos[i, 1]
        y_min = y - d[0, i]
        y_max = y + d[1, i]
        x_min = x - d[2, i]
        x_max = x + d[3, i]
        rotate_mat = get_rotate_mat(-angle[i])

        temp_x = np.array([[x_min, x_max, x_max, x_min]]) - x
        temp_y = np.array([[y_min, y_min, y_max, y_max]]) - y
        coordinates = np.concatenate((temp_x, temp_y), axis=0)
        res = np.dot(rotate_mat, coordinates)
        res[0, :] += x
        res[1, :] += y

        if is_valid_poly(res, score_shape, scale):
            index.append(i)
            polys.append([res[0, 0], res[1, 0], res[0, 1], res[1, 1], res[0, 2], res[1, 2], res[0, 3], res[1, 3]])
    return np.array(polys), index


def get_boxes(confidence, geometries, confidence_thresh=0.9, nms_thresh=0.2):
    """Get boxes from feature map"""

    confidence = confidence[0, :, :]
    xy_text = np.argwhere(confidence > confidence_thresh)  # n x 2, format is [r, c]
    if xy_text.size == 0:
        return None

    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    valid_pos = xy_text[:, ::-1].copy()  # n x 2, [x, y]
    valid_geometries = geometries[:, xy_text[:, 0], xy_text[:, 1]]  # 5 x n
    polys_restored, index = restore_polys(valid_pos, valid_geometries, confidence.shape)
    if polys_restored.size == 0:
        return None

    boxes = np.zeros((polys_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = polys_restored
    boxes[:, 8] = confidence[xy_text[index, 0], xy_text[index, 1]]
    boxes = lanms.merge_quadrangle_n9(boxes.astype("float32"), nms_thresh)
    return boxes


def adjust_ratio(boxes, ratio_w, ratio_h):
    """Refine boxes"""

    if boxes is None or boxes.size == 0:
        return None
    boxes[:, [0, 2, 4, 6]] /= ratio_w
    boxes[:, [1, 3, 5, 7]] /= ratio_h
    return np.around(boxes)  # rounds to nearest even value


def detect(image, model, device):
    """Detect text regions of image using model"""

    model.eval()
    image, ratio_h, ratio_w = resize(image)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
    )
    image = transform(image)
    image = torch.unsqueeze(image, 0).to(device)

    with torch.no_grad():
        confidence, geometries = model(image)

    confidence = torch.squeeze(confidence, 0).cpu().numpy()
    geometries = torch.squeeze(geometries, 0).cpu().numpy()
    boxes = get_boxes(confidence, geometries)

    return adjust_ratio(boxes, ratio_w, ratio_h)


def plot_boxes(image, boxes):
    """Plot quadrangles on image"""
    if boxes is None:
        return image

    draw = ImageDraw.Draw(image)
    for box in boxes:
        coordinates = [box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]]
        draw.polygon(coordinates, outline=(0, 255, 0))

    return image


def detect_dataset(model, device, test_img_path, submit_path):
    """Detection on whole dataset, save .txt results in submit_path"""

    filenames = os.listdir(test_img_path)
    path2filenames = sorted([os.path.join(test_img_path, filename) for filename in filenames])

    for idx, path2filename in enumerate(path2filenames):
        print("Evaluating {} image".format(idx), end="\r")
        boxes = detect(Image.open(path2filename), model, device)
        seq = []
        if boxes is not None:
            seq.extend([",".join([str(int(b)) for b in box[:-1]]) + "\n" for box in boxes])

        path = os.path.join(submit_path, "res_" + os.path.basename(path2filename).replace(".jpg", ".txt"))
        with open(path, "w") as f:
            f.writelines(seq)


def parse_opt():
    parser = argparse.ArgumentParser(description="EAST inference arguments")
    parser.add_argument("--cfg", default="D", help="Configuration for backbone VGG (default D, [A, B, D, E)")
    parser.add_argument("--weights", default="weights/model.pt", help="Path to weight file (default: model.pt)")
    parser.add_argument("--input", type=str, default="data/ch4_test_images/img_10.jpg", help="Path to input image")
    parser.add_argument("--output", default="output.jpg", help="Path to save mask image")

    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize and Load weights
    model = EAST(cfg=opt.cfg).to(device)
    model.load_state_dict(torch.load(opt.weights))
    model.float()

    # Read image
    image = Image.open(opt.input)

    # Inference
    boxes = detect(image, model, device)
    plot_img = plot_boxes(image, boxes)

    # Save results
    plot_img.save(opt.output)
