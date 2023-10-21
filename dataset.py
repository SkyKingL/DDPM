import os

from PIL import Image
from torch.utils.data import Dataset


class LSUNChurchDataset(Dataset):
    def __init__(self, data_dir="church_outdoor_train", image_size=128, transform=None):
        super(LSUNChurchDataset, self).__init__()
        self.data_dir = data_dir
        self.image_size = image_size
        self.image_files = self._load_image_files()
        self.transform = transform  # 图片预处理的变换

    def _load_image_files(self):
        image_files = []
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith(".jpg"):
                image_files.append(os.path.join(self.data_dir, file_name))
        return image_files  # string_path list

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_file = self.image_files[index]  # index指向的文件名
        return self.transform(Image.open(image_file)), image_file
