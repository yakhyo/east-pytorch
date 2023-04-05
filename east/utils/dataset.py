import os

from PIL import Image
from torch.utils import data
from torchvision import transforms

from .misc import adjust_height, crop, extract_vertices, get_score_geo, rotate


def create_dataloader(data_path, batch_size, num_workers):
    dataset = Dataset(data_path)
    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True
    )
    return loader


class Dataset(data.Dataset):
    def __init__(self, data_path, scale=0.25, length=512):
        super().__init__()
        image_path = data_path
        gt_path = image_path.replace("_images", "_gt")
        self.image_files = [os.path.join(image_path, image_file) for image_file in sorted(os.listdir(image_path))]
        self.gt_files = [os.path.join(gt_path, gt_file) for gt_file in sorted(os.listdir(gt_path))]
        self.scale = scale
        self.length = length
        self.transform = transforms.Compose(
            [
                transforms.ColorJitter(0.5, 0.5, 0.5, 0.25),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        with open(self.gt_files[index], "r") as f:
            lines = f.readlines()
        vertices, labels = extract_vertices(lines)

        image = Image.open(self.image_files[index])
        image, vertices = adjust_height(image, vertices)
        image, vertices = rotate(image, vertices)
        image, vertices = crop(image, vertices, labels, self.length)
        score_map, geo_map, ignored_map = get_score_geo(image, vertices, labels, self.scale, self.length)

        image = self.transform(image)

        return image, score_map, geo_map, ignored_map
