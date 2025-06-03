import json
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split


class CatDataset(Dataset):
    def __init__(self, image_paths, labels, root_dir, transform=None):
        """
        构造函数
        :param image_paths: 图片存放路径
        :param labels: 标签
        :param root_dir: 图片数据的根目录
        :param transform: 数据增强策略
        """
        self.image_paths = image_paths
        self.labels = labels
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        # 相对路径
        rel_path = self.image_paths[item]
        label = self.labels[item]

        # 拼接成完整路径
        full_path = os.path.join(self.root_dir, rel_path)

        image = Image.open(full_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label


def load_datasets_from_json(json_path, image_root_dir, val_ratio=0.2, test_ratio=0.1):
    """
    从json文件中加载数据集
    :param json_path: json文件路径
    :param image_root_dir: 图片文件根目录
    :param val_ratio: 验证集比例
    :param test_ratio: 测试集比例
    :return: 训练集，验证集，测试集
    """

    # 加载json文件
    with open(json_path, "r") as f:
        all_labels = json.load(f)

    image_paths = list(all_labels.keys())
    labels = [all_labels[path] for path in image_paths]

    # 分离测试集
    X_temp, X_test, y_temp, y_test = train_test_split(image_paths, labels, test_size=test_ratio, stratify=labels,
                                                      random_state=42)

    # 分离验证集
    val_size = val_ratio / (1 - test_ratio)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size, stratify=y_temp,
                                                      random_state=42)

    # 定义训练集数据增强策略
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        # 随机旋转
        transforms.RandomRotation(10),
        # 随机水平翻转
        transforms.RandomHorizontalFlip(),
        # 调整色彩
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        # 使用ImageNet数据集的均值和方差进行归一化
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 定义验证集和测试集数据增强策略
    transform_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # 构建数据集对象
    train_dataset = CatDataset(X_train, y_train, image_root_dir, transform_train)
    val_dataset = CatDataset(X_val, y_val, image_root_dir, transform_eval)
    test_dataset = CatDataset(X_test, y_test, image_root_dir, transform_eval)

    return train_dataset, val_dataset, test_dataset
