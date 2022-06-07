# Butterfly-Recognition-System
以残差网络（ResNet）为核心的蝴蝶识别系统

项目目录结构（库中文件全部放在根目录下）
├─static
│  ├─html_resource       ##该文件夹下存放所有模板需要的资源
│  │  └─butterfly_photo    ##该文件夹存放20类蝴蝶的典型图片
│  └─resources             
│      └─received_images   ##存放从前端加载的图片
├─templates              ##该文件夹下存放Flask模板文件（html）
├─__pycache__
├─.idea
│  └─inspectionProfiles
├─butterfly              ##存放数据集（20类分别存在20个文件夹中，文件夹名为类名）
├─migrations
│  ├─versions
│  │  └─__pycache__
│  └─__pycache__
├─upload
└─.static
    └─resources          
        └─received_images  

1.在butterfly文件夹中放好数据（20类分别放在20个文件夹中）
2.运行butterfly.py在butterfly文件夹中生成一个image.csv文件
3.运行train_scratch.py文件，训练模型
4.运行predict.py文件，输入要识别的图片路径即可得到识别结果。
5.运行Sever.py文件可以在系统中识别。
