import os
from PIL import Image
from torch.utils import data
import torchvision.transforms as transforms

from util.utils import rotate_img, extract_vertices, adjust_height, crop_img, get_score_geo


class EASTDataset(data.Dataset):
    def __init__(self, img_path, gt_path, scale=0.25, length=512):
        super(EASTDataset, self).__init__()
        self.img_files = [os.path.join(img_path, img_file) for img_file in sorted(os.listdir(img_path))]
        self.gt_files = [os.path.join(gt_path, gt_file) for gt_file in sorted(os.listdir(gt_path))]
        self.scale = scale
        self.length = length
        self.mean = (0.5, 0.5, 0.5)
        self.std = (0.5, 0.5, 0.5)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        with open(self.gt_files[index], 'r') as f:
            lines = f.readlines()
        vertices, labels = extract_vertices(lines)

        img = Image.open(self.img_files[index])
        img, vertices = adjust_height(img, vertices)
        img, vertices = rotate_img(img, vertices)
        img, vertices = crop_img(img, vertices, labels, self.length)

        transform = transforms.Compose([
            transforms.ColorJitter(0.5, 0.5, 0.5, 0.25),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)])

        score_map, geo_map, ignored_map = get_score_geo(img, vertices, labels, self.scale, self.length)
        return transform(img), score_map, geo_map, ignored_map
