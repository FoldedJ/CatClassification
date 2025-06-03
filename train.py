import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from PetDataset import load_datasets_from_json
import importlib


# 读取配置文件
with open("config.json", "r") as f:
    config = json.load(f)

model_module = config["model_module"]
model_name = config["model_name"]
json_path = config["json_path"]
image_root_dir = config["image_root_dir"]
num_classes = config["num_classes"]
batch_size = config["batch_size"]
epochs = config["epochs"]
learning_rate = config["learning_rate"]
save_path = config["save_path"]
device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

# 加载数据集
train_dataset, val_dataset, test_dataset = load_datasets_from_json(
    json_path=json_path,
    image_root_dir=image_root_dir
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=True
)

# 构造模型
module = importlib.import_module(model_module)
build_model_fn = getattr(module, model_name)
model = build_model_fn(num_classes=num_classes).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练
best_val_acc = 0.0
best_epoch = 0
# 早停轮数
early_stopping = 5
# 没有提升的轮数
no_improvement_count = 0

print(f"开始训练，共 {epochs} 个epoch，使用设备: {device}")

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        # 统计
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += targets.size(0)
        train_correct += predicted.eq(targets).sum().item()

        if (batch_idx + 1) % 5 == 0:
            progress = (batch_idx + 1) / len(train_loader) * 100
            print(f"Epoch: {epoch} | Batch: {batch_idx + 1}/{len(train_loader)} [{progress:.1f} | Loss: {loss.item():.4f} | Acc: {100. * train_correct / train_total:.2f}%")

    train_acc = 100. * train_correct / train_total
    train_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch}, Train data : Loss={train_loss:.4f}, Acc={train_acc:.2f}%")

    # 验证
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    # 禁止梯度更新
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 统计
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()

    val_acc = 100. * val_correct / val_total
    val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch}, Validation data: Loss={val_loss:.4f}, Acc={val_acc:.2f}%")

    # 保存最佳模型
    if val_acc > best_val_acc:
        torch.save(model.state_dict(), save_path)
        best_val_acc = val_acc
        best_epoch = epoch
        no_improvement_count = 0
        print(f"验证集准确率提高: {best_val_acc:.2f}% -> {val_acc:.2f}%，保存模型到 {save_path}")
    else:
        no_improvement_count += 1
        print(f"验证集准确率未提高，已连续 {no_improvement_count} 个epoch")

    # 早停机制
    if no_improvement_count >= early_stopping:
        print(f"早停触发: 验证集准确率在 {early_stopping} 个epoch内未提高")
        break

print(f"训练完成！最佳模型在epoch {best_epoch}，验证集准确率: {best_val_acc:.2f}%")
