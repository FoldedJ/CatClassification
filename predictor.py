import torch
import json
import importlib
from torchvision import transforms
from PIL import Image
import os


with open("config.json", "r") as f:
    config = json.load(f)

model_module = config["model_module"]
model_name = config["model_name"]
json_path = config["json_path"]
num_classes = config["num_classes"]
batch_size = config["batch_size"]
learning_rate = config["learning_rate"]
save_path = config["save_path"]
device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

class_names = ["阿比西尼亚猫", "孟买猫", "埃及猫", "异国短毛猫", "喜马拉雅猫", "缅因猫", "布偶猫", "俄罗斯蓝猫", "苏格兰折耳猫", "暹罗猫", "无毛猫"]

# 构建模型
module = importlib.import_module(model_module)
build_model_fn = getattr(module, model_name)
model = build_model_fn(num_classes=num_classes)

# 加载模型
model = model.to(device)
model.load_state_dict(torch.load(save_path))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image_path):
    """预测单张图片的猫品种"""
    try:
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        # 预处理
        input_tensor = transform(image).unsqueeze(0)  # 添加批次维度
        input_tensor = input_tensor.to(device)

        # 模型推理
        with torch.no_grad():
            outputs = model(input_tensor)
            # 获取预测概率和类别
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

        # 返回预测结果
        predicted_class = class_names[pred.item()]
        confidence = conf.item() * 100  # 转换为百分比

        return predicted_class, confidence

    except Exception as e:
        print(f"预测过程中发生错误: {e}")
        return None, None

if __name__ == "__main__":
    # 获取用户输入的图片路径
    image_path = input("请输入要预测的图片路径: ")

    # 检查图片路径是否存在
    if not os.path.exists(image_path):
        print(f"错误: 图片路径不存在 - {image_path}")
    else:
        # 进行预测
        breed, confidence = predict_image(image_path)

        # 输出结果
        if breed and confidence:
            print(f"\n{'=' * 40}")
            print(f"预测结果: {breed}")
            print(f"置信度: {confidence:.2f}%")
            print(f"{'=' * 40}")