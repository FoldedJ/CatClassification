import os
import json

data_dir = "dataset"
output_file = "label.json"

class_names = sorted(os.listdir(data_dir))
class_to_idx = {cls_name : idx for idx, cls_name in enumerate(class_names)}

labels = {}

for cls_name in class_names:
    cls_folder = os.path.join(data_dir, cls_name)
    for fname in os.listdir(cls_folder):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(cls_name, fname)
            labels[file_path] = class_to_idx[cls_name]

with open(output_file, "w") as f :
    json.dump(labels, f, indent=2)

print(f"已保存{len(labels)} 个数据至 {output_file}")
