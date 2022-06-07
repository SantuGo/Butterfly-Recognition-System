import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from resnet34 import ResNet34


def predict():
    # 设置设备为显卡
    device = torch.device('cuda')

    # 根据路径获取图片
    img_path = input("请输入需要预测图片的路径:")
    # img_path = "dataset/用于验证准确率的图片/CLOUDED SULPHUR/4.jpg"
    assert os.path.exists(img_path), f"file {img_path} dose not exist."
    img = Image.open(img_path)
    plt.imshow(img)

    # 图片预处理
    data_transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]
    )
    img = data_transform(img)
    # [C, H, W] -> [1, C, H, W]
    img = torch.unsqueeze(img, dim=0)

    # 获取json文件
    json_path = './class_indices.json'
    assert os.path.exists(json_path), f"file {json_path} does not exist."
    json_file = open(json_path, 'r')
    class_indict = json.load(json_file)

    # 设置模型骨架并加载已经训练好的模型参数
    model = ResNet34(num_classes=20).to(device)
    model_weight_path = "best_model_weight.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    #预测
    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
    # print(predict_cla)
    # 打印预测结果并显示置信度
    print_res = "class: {}  prob: {:.3f}".format(class_indict[str(predict_cla)], predict[predict_cla].numpy())

    plt.title(print_res)
    print(print_res)
    plt.show()


if __name__ == '__main__':
    predict()