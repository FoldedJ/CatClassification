import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PetDataset import load_datasets_from_json
import json
import importlib

with open("config.json", "r") as f:
    config = json.load(f)

model_module = config["model_module"]
model_name = config["model_name"]
json_path = config["json_path"]
num_classes = config["num_classes"]
batch_size = config["batch_size"]
learning_rate = config["learning_rate"]
save_path = config["save_path"]
image_root_dir = config["image_root_dir"]
device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

# 构建模型
module = importlib.import_module(model_module)
build_model_fn = getattr(module, model_name)
model = build_model_fn(num_classes=num_classes)

# 加载模型
model = model.to(device)
model.load_state_dict(torch.load(save_path))

# 加载数据集
train_dataset, val_dataset, test_dataset = load_datasets_from_json(
    json_path=json_path,
    image_root_dir=image_root_dir
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True
)

criterion = nn.CrossEntropyLoss()

test_loss = 0.0
test_correct = 0
test_total = 0

# 测试阶段
model.eval()

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 统计
        test_loss += loss.item()
        _, predicted = outputs.max(1)
        test_total += targets.size(0)
        test_correct += predicted.eq(targets).sum().item()

test_loss = test_loss / len(test_loader)
test_acc = 100. * test_correct / test_total

print(f"测试集平均损失: {test_loss:.4f}")
print(f"测试集准确率: {test_acc:.2f}%")
