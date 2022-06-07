# 导入常用的库
import time
import os

import requests
import torch
from PIL import Image
import torchvision.transforms as transforms
import json

# 导入数据库相关文件
from werkzeug.utils import secure_filename

import Setting
from flask_sqlalchemy import SQLAlchemy
from exts import db
from Models import AdminModel
from forms import LoginForm, RegisterForm

# 导入flask库的Flask类和request对象
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from flask_migrate import Migrate
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config.from_object(Setting)

# ------------------------------------------------------1.加载模型---------------------------------------------------------
path_model = "best_model.pth"
model_loaded = torch.load(path_model)

device = torch.device('cuda')

# 获取json文件
json_path = './class_indices.json'
assert os.path.exists(json_path), f"file {json_path} does not exist."
json_file = open(json_path, 'r')
class_indict = json.load(json_file)


# ------------------------------------------------------2.获取测试图片-----------------------------------------------------
# 根据图片文件路径获取图像数据矩阵
def get_image(imageFilePath):
    img = Image.open(imageFilePath)
    return img


# ------------------------------------------------------3.定义图片预处理--------------------------------------------------------------
# 模型预测前必要的图像处理
def preprocess_image(input_image):
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    ])
    # img_chw = preprocess(input_image)
    img = preprocess(input_image)
    img = torch.unsqueeze(img, dim=0)
    # [C, H, W] -> [1, C, H, W]三维转四维，一次输入的只有一张图片，与神经网络的输入格式相匹配
    # return img_chw  # chw:channel height width
    return img


# ------------------------------------------------------4.模型预测--------------------------------------------------------------
# 使用模型对指定图片文件路径完成图像分类，返回值为预测的种类名称
def predict_image(model, imageFilePath):
    model.eval()  # 参数固化
    input_image = get_image(imageFilePath)
    img = preprocess_image(input_image)
    if torch.cuda.is_available():
        img = img.to('cuda')
        model.to('cuda')
    # input_list = [img]

    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
        #去掉维数为1的的维度torch.squeeze()
        # 输入图片类型:ATALA 对应下标:5
        # print(torch.squeeze(model(img.to(device))))
        # tensor([-0.7175,  4.5696,  1.1368, -1.6864,  6.7974, 12.5798, -0.7490,  3.1468,
        #          1.0797, -3.8273, -6.0243, -3.5864, -2.1185, -0.7268,  2.9118, -7.8169,
        #         -0.0261, -0.7262, -0.3111, -2.5701], device='cuda:0')

        # print(output)
        # tensor([-0.7175, 4.5696, 1.1368, -1.6864, 6.7974, 12.5798, -0.7490, 3.1468,
        #         1.0797, -3.8273, -6.0243, -3.5864, -2.1185, -0.7268, 2.9118, -7.8169,
        #         -0.0261, -0.7262, -0.3111, -2.5701])

        predict = torch.softmax(output, dim=0)  # 按列进行softmax,列和为1
        # print(predict)
        # tensor([1.6731e-06, 3.3087e-04, 1.0686e-05, 6.3493e-07, 3.0704e-03, 9.9642e-01,
        #         1.6211e-06, 7.9753e-05, 1.0093e-05, 7.4637e-08, 8.2950e-09, 9.4967e-08,
        #         4.1216e-07, 1.6576e-06, 6.3049e-05, 1.3813e-09, 3.3404e-06, 1.6586e-06,
        #         2.5119e-06, 2.6239e-07])

        # predict_cla = torch.argmax(predict)
        # print(predict_cla)
        # tensor(5)

        # print(predict_cla)
        predict_cla = torch.argmax(predict).numpy()
        # print(predict_cla)
        # ALATA

    # 打印预测结果并显示置信度
    print_res1 = "蝴蝶种类: {}".format(class_indict[str(predict_cla)])
    # print(print_res1)
    print_res2 = "可信度: {:.3f}".format(predict[predict_cla].numpy())
    result_cls = class_indict[str(predict_cla)]
    return print_res1, print_res2, result_cls


# ------------------------------------------------------5.服务返回--------------------------------------------------------------

# 连接数据库
db.init_app(app)
migrate = Migrate(app, db)


# 访问首页时的调用函数
@app.route('/')
def index_page():
    # engine = db.get_engine()                ##测试数据库连接是否正常
    # with engine.connect() as conn:
    #     result = conn.execute("select 1")
    #     print(result.fetchone())           ##打印  (1,)  连接正常

    return render_template('index.html')


# 点击"点击进入"直接跳转至预测页面
@app.route('/predict')
def predict_page():
    return render_template('predict.html')


# # 管理员注册
# @app.route("/admin_register", methods=['POST', 'GET'])
# def admin_register():
#     if request.method == 'GET':
#         return render_template("admin_register.html")
#     else:
#         form = RegisterForm(request.form)
#         if form.validate():
#             admin_registername = form.admin_registername.data
#             register_password = form.register_password.data
#             hash_password = generate_password_hash(register_password)
#
#             admin = AdminModel(admin_name = admin_registername, password = hash_password)
#             db.session.add(admin)
#             db.session.commit()
#             return redirect("admin_login")
#
#         else:
#             return "账号或密码格式错误！"

# 管理员登录
@app.route("/admin_login", methods=['POST', 'GET'])
def admin_login():
    if request.method == 'GET':
        return render_template("admin_login.html")
    else:
        form = LoginForm(request.form)
        if form.validate():
            admin_name = form.admin_name.data
            password = form.password.data
            admin = AdminModel.query.filter_by(admin_name=admin_name).first()
            # if admin and admin.password == password:
            if admin and check_password_hash(admin.password, password):
                session['admin_id'] = admin.id
                return redirect(url_for('admin_page'))
            else:
                flash("用户名或密码有误，请重新输入！")
                return redirect('/admin_login')
                # result = '用户名或密码有误，请重新输入！'
                # return render_template("admin_login.html", result=result)
        else:
            flash("用户名或密码格式错误，请重新输入！")
            return redirect('/admin_login')
            # result = '账号或密码格式错误！'
            # return render_template("admin_login.html", result=result)


@app.route("/admin_page", methods=['POST', 'GET'])
def admin_page():
        return render_template("admin_page.html")


# 使用predict_image这个API服务时的调用函数
@app.route("/upload_image", methods=['POST', 'GET'])
def predict():
    startTime = time.time()
    global received_file
    received_file = request.files['input_image']
    # print("1 "+received_file)

    global imageFileName
    imageFileName = received_file.filename
    # 获取前缀（文件名称）print(os.path.splitext(imageFileName)[0])
    # 获取后缀（文件类型）print(os.path.splitext(imageFileName)[-1])
    if received_file:
        received_dirPath = 'static/resources/received_images'
        if not os.path.isdir(received_dirPath):
            os.makedirs(received_dirPath)
        global imageFilePath
        imageFilePath = os.path.join(received_dirPath, imageFileName)
        # print(imageFilePath)
        # print(type(imageFilePath))  #<class 'str'>

        if os.path.exists(imageFilePath):
            # print("存在了")
            # print(type(imageFileName))  #<class 'str'>
            imageFileName = '0' + imageFileName
            # print("imageFileName="+imageFileName)
        imageFilePath = os.path.join(received_dirPath, imageFileName)
        # print("已修改文件名")

        # for file in os.listdir(received_dirPath):
        #     name = file.split('.')[0]
        #     os.rename(os.path.join(path, file), os.path.join(path, '%05d' % int(name) + ".txt"))  # ‘%05d’表示一共5位数
        # global result_cls
        # result1, result2, result_cls = predict_image(model_loaded, imageFilePath)
        # preprocess_image(received_file)
        received_file.save(imageFilePath)
        print('image file saved to %s' % imageFilePath)
        usedTime = time.time() - startTime
        print('接收图片并保存，总共耗时%.2f秒' % usedTime)
        startTime = time.time()
        global result_cls
        result1, result2, result_cls = predict_image(model_loaded, imageFilePath)
        imageFilePath = imageFilePath.replace("\\", "/")
        imageFilePath = "" + imageFilePath
        print("预测原图路径：" + imageFilePath)
        # print(result)
        # print("预测类名为："+result_cls) #----> 预测类名
        result_cls_web = result_cls.replace(' ', '_')
        result_cls_web = '../static/html_resource/butterfly_photo/' + result_cls_web + '.jpg'
        # print("典型图片地址为：" + result_cls)
        result1 = str(result1)
        result2 = str(result2)
        usedTime = time.time() - startTime
        print('完成对接收图片的预测，总共耗时%.2f秒' % usedTime)
        # return result
        return render_template("result.html", result1=result1, result2=result2, result_cls=result_cls_web,
                               imageFilePath=imageFilePath)
    else:
        return render_template("fail.html")


@app.route("/result", methods=['POST', 'GET'])
def route():
    return


# 判断结果时返回index.html
@app.route("/true", methods=['POST', 'GET'])
def true():
    saveFileName = imageFileName  # 文件名为  1.jpg
    # print(saveFileName)  #1.jpg  在predict方法，即第一次将预测图片加入received_image文件夹后图片有可能已经加过一次0了
    saved_dirPath = '.\\butterfly\\' + result_cls
    print("要保存到数据集的位置是：" + saved_dirPath)
    saveFilePath = os.path.join(saved_dirPath, saveFileName)
    # print(saveFilePath)   #.\butterfly\BLUE MORPHO\1.jpg
    if os.path.exists(saveFilePath):
        saveFileName = '0' + saveFileName

    saveFilePath = os.path.join(saved_dirPath, saveFileName)
    img = Image.open(imageFilePath)
    img.save(saveFilePath)

    return render_template("predict.html")


@app.route("/false", methods=['POST', 'GET'])
def false():
    return render_template("select_true.html")

@app.route("/false/select_true", methods=['POST', 'GET'])
def select_true():
    path = request.form["true_class"]
    saveFileName = imageFileName  # 文件名为  1.jpg
    # print(saveFileName)  #1.jpg  在predict方法，即第一次将预测图片加入received_image文件夹后图片有可能已经加过一次0了
    saved_dirPath = '.\\butterfly\\' + path
    print("要保存到数据集的位置是：" + saved_dirPath)
    saveFilePath = os.path.join(saved_dirPath, saveFileName)
    # print(saveFilePath)   #.\butterfly\BLUE MORPHO\1.jpg
    if os.path.exists(saveFilePath):
        saveFileName = '0' + saveFileName

    saveFilePath = os.path.join(saved_dirPath, saveFileName)
    img = Image.open(imageFilePath)
    img.save(saveFilePath)
    return redirect('/predict')



@app.route("/unknown", methods=['POST', 'GET'])
def unknown():
    return render_template("predict.html")



# 管理员更新模型
@app.route("/update_model", methods=['POST', 'GET'])
def update_model():
    # content1 = "更新模型大概需要30min~60min，请耐心等待！"
    # content2 = "请勿关闭当前页面，模型更新完毕后自动返回预测页面！"
    # os.system('python butterfly.py')  # 执行csv文件更新操作
    # os.system('python train_scratch.py')
    # os.system('python channel_image.py')
    return redirect("/predict")


# 管理员添加数据
# 验证格式是否正确
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


@app.route("/select_class", methods=['POST', 'GET'])
def select_class():
    return render_template("select_class.html")


@app.route("/import_data_ADONIS", methods=['POST', 'GET'])
def import_data_ADONIS():
    app.config['UPLOAD_FOLDER'] = 'butterfly/ADONIS/'
    uploaded_files = request.files.getlist("file[]")
    filenames = []
    for file in uploaded_files:
        # Check if the file is one of the allowed types/extensions
        if file and allowed_file(file.filename):
            # Make the filename safe, remove unsupported chars
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filenames.append(filename)
    return redirect('/select_class')


@app.route("/import_data_AFRICAN_GIANT_SWALLOWTAIL", methods=['POST', 'GET'])
def import_data_AFRICAN_GIANT_SWALLOWTAIL():
    app.config['UPLOAD_FOLDER'] = 'butterfly/AFRICAN GIANT SWALLOWTAIL/'
    uploaded_files = request.files.getlist("file[]")
    filenames = []
    for file in uploaded_files:
        # Check if the file is one of the allowed types/extensions
        if file and allowed_file(file.filename):
            # Make the filename safe, remove unsupported chars
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filenames.append(filename)
    return redirect('/select_class')


@app.route("/import_data_AMERICAN_SNOOT", methods=['POST', 'GET'])
def import_data_AMERICAN_SNOOT():
    app.config['UPLOAD_FOLDER'] = 'butterfly/AMERICAN SNOOT'
    uploaded_files = request.files.getlist("file[]")
    filenames = []
    for file in uploaded_files:
        # Check if the file is one of the allowed types/extensions
        if file and allowed_file(file.filename):
            # Make the filename safe, remove unsupported chars
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filenames.append(filename)
    return redirect('/select_class')


@app.route("/import_data_AN_88", methods=['POST', 'GET'])
def import_data_AN_88():
    app.config['UPLOAD_FOLDER'] = 'butterfly/AN 88'
    uploaded_files = request.files.getlist("file[]")
    filenames = []
    for file in uploaded_files:
        # Check if the file is one of the allowed types/extensions
        if file and allowed_file(file.filename):
            # Make the filename safe, remove unsupported chars
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filenames.append(filename)
    return redirect('/select_class')


@app.route("/import_data_APPOLLO", methods=['POST', 'GET'])
def import_data_APPOLLO():
    app.config['UPLOAD_FOLDER'] = 'butterfly/APPOLLO'
    uploaded_files = request.files.getlist("file[]")
    filenames = []
    for file in uploaded_files:
        # Check if the file is one of the allowed types/extensions
        if file and allowed_file(file.filename):
            # Make the filename safe, remove unsupported chars
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filenames.append(filename)
    return redirect('/select_class')


@app.route("/import_data_ATALA", methods=['POST', 'GET'])
def import_data_ATALA():
    app.config['UPLOAD_FOLDER'] = 'butterfly/ATALA'
    uploaded_files = request.files.getlist("file[]")
    filenames = []
    for file in uploaded_files:
        # Check if the file is one of the allowed types/extensions
        if file and allowed_file(file.filename):
            # Make the filename safe, remove unsupported chars
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filenames.append(filename)
    return redirect('/select_class')


@app.route("/import_data_BANDED_ORANGE_HELICONIAN", methods=['POST', 'GET'])
def import_data_BANDED_ORANGE_HELICONIAN():
    app.config['UPLOAD_FOLDER'] = 'butterfly/BANDED ORANGE HELICONIAN'
    uploaded_files = request.files.getlist("file[]")
    filenames = []
    for file in uploaded_files:
        # Check if the file is one of the allowed types/extensions
        if file and allowed_file(file.filename):
            # Make the filename safe, remove unsupported chars
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filenames.append(filename)
    return redirect('/select_class')


@app.route("/import_data_BANDED_PEACOCK", methods=['POST', 'GET'])
def import_data_BANDED_PEACOCK():
    app.config['UPLOAD_FOLDER'] = 'butterfly/BANDED PEACOCK'
    uploaded_files = request.files.getlist("file[]")
    filenames = []
    for file in uploaded_files:
        # Check if the file is one of the allowed types/extensions
        if file and allowed_file(file.filename):
            # Make the filename safe, remove unsupported chars
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filenames.append(filename)
    return redirect('/select_class')


@app.route("/import_data_BECKERS_WHITE", methods=['POST', 'GET'])
def import_data_BECKERS_WHITE():
    app.config['UPLOAD_FOLDER'] = 'butterfly/BECKERS WHITE'
    uploaded_files = request.files.getlist("file[]")
    filenames = []
    for file in uploaded_files:
        # Check if the file is one of the allowed types/extensions
        if file and allowed_file(file.filename):
            # Make the filename safe, remove unsupported chars
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filenames.append(filename)
    return redirect('/select_class')


@app.route("/import_data_BLACK_HAIRSTREAK", methods=['POST', 'GET'])
def import_data_BLACK_HAIRSTREAK():
    app.config['UPLOAD_FOLDER'] = 'butterfly/BLACK HAIRSTREAK'
    uploaded_files = request.files.getlist("file[]")
    filenames = []
    for file in uploaded_files:
        # Check if the file is one of the allowed types/extensions
        if file and allowed_file(file.filename):
            # Make the filename safe, remove unsupported chars
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filenames.append(filename)
    return redirect('/select_class')


@app.route("/import_data_BLUE_MORPHO", methods=['POST', 'GET'])
def import_data_BLUE_MORPHO():
    app.config['UPLOAD_FOLDER'] = 'butterfly/BLUE MORPHO'
    uploaded_files = request.files.getlist("file[]")
    filenames = []
    for file in uploaded_files:
        # Check if the file is one of the allowed types/extensions
        if file and allowed_file(file.filename):
            # Make the filename safe, remove unsupported chars
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filenames.append(filename)
    return redirect('/select_class')


@app.route("/import_data_BLUE_SPOTTED_CROW", methods=['POST', 'GET'])
def import_data_BLUE_SPOTTED_CROW():
    app.config['UPLOAD_FOLDER'] = 'butterfly/BLUE SPOTTED CROW'
    uploaded_files = request.files.getlist("file[]")
    filenames = []
    for file in uploaded_files:
        # Check if the file is one of the allowed types/extensions
        if file and allowed_file(file.filename):
            # Make the filename safe, remove unsupported chars
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filenames.append(filename)
    return redirect('/select_class')


@app.route("/import_data_BROWN_SIPROETA", methods=['POST', 'GET'])
def import_data_BROWN_SIPROETA():
    app.config['UPLOAD_FOLDER'] = 'butterfly/BROWN SIPROETA'
    uploaded_files = request.files.getlist("file[]")
    filenames = []
    for file in uploaded_files:
        # Check if the file is one of the allowed types/extensions
        if file and allowed_file(file.filename):
            # Make the filename safe, remove unsupported chars
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filenames.append(filename)
    return redirect('/select_class')


@app.route("/import_data_CABBAGE_WHITE", methods=['POST', 'GET'])
def import_data_CABBAGE_WHITE():
    app.config['UPLOAD_FOLDER'] = 'butterfly/CABBAGE WHITE'
    uploaded_files = request.files.getlist("file[]")
    filenames = []
    for file in uploaded_files:
        # Check if the file is one of the allowed types/extensions
        if file and allowed_file(file.filename):
            # Make the filename safe, remove unsupported chars
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filenames.append(filename)
    return redirect('/select_class')


@app.route("/import_data_CAIRNSBIRDWING", methods=['POST', 'GET'])
def import_data_CAIRNSBIRDWING():
    app.config['UPLOAD_FOLDER'] = 'butterfly/CAIRNS BIRDWING'
    uploaded_files = request.files.getlist("file[]")
    filenames = []
    for file in uploaded_files:
        # Check if the file is one of the allowed types/extensions
        if file and allowed_file(file.filename):
            # Make the filename safe, remove unsupported chars
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filenames.append(filename)
    return redirect('/select_class')


@app.route("/import_data_CHECQUERED_SKIPPER", methods=['POST', 'GET'])
def import_data_CHECQUERED_SKIPPER():
    app.config['UPLOAD_FOLDER'] = 'butterfly/CHECQUERED SKIPPER'
    uploaded_files = request.files.getlist("file[]")
    filenames = []
    for file in uploaded_files:
        # Check if the file is one of the allowed types/extensions
        if file and allowed_file(file.filename):
            # Make the filename safe, remove unsupported chars
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filenames.append(filename)
    return redirect('/select_class')


@app.route("/import_data_CHESTNUT", methods=['POST', 'GET'])
def import_data_CHESTNUT():
    app.config['UPLOAD_FOLDER'] = 'butterfly/CHESTNUT'
    uploaded_files = request.files.getlist("file[]")
    filenames = []
    for file in uploaded_files:
        # Check if the file is one of the allowed types/extensions
        if file and allowed_file(file.filename):
            # Make the filename safe, remove unsupported chars
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filenames.append(filename)
    return redirect('/select_class')


@app.route("/import_data_CLEOPATRA", methods=['POST', 'GET'])
def import_data_CLEOPATRA():
    app.config['UPLOAD_FOLDER'] = 'butterfly/CHESTNUT'
    uploaded_files = request.files.getlist("file[]")
    filenames = []
    for file in uploaded_files:
        # Check if the file is one of the allowed types/extensions
        if file and allowed_file(file.filename):
            # Make the filename safe, remove unsupported chars
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filenames.append(filename)
    return redirect('/select_class')


@app.route("/import_data_CLOUDED_SULPHUR", methods=['POST', 'GET'])
def import_data_CLOUDED_SULPHUR():
    app.config['UPLOAD_FOLDER'] = 'butterfly/CLOUDED SULPHUR'
    uploaded_files = request.files.getlist("file[]")
    filenames = []
    for file in uploaded_files:
        # Check if the file is one of the allowed types/extensions
        if file and allowed_file(file.filename):
            # Make the filename safe, remove unsupported chars
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filenames.append(filename)
    return redirect('/select_class')


@app.route("/import_data_CRECENT", methods=['POST', 'GET'])
def import_data_CRECENT():
    app.config['UPLOAD_FOLDER'] = 'butterfly/CRECENT'
    uploaded_files = request.files.getlist("file[]")
    filenames = []
    for file in uploaded_files:
        # Check if the file is one of the allowed types/extensions
        if file and allowed_file(file.filename):
            # Make the filename safe, remove unsupported chars
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filenames.append(filename)
    return redirect('/select_class')


# 主函数
if __name__ == "__main__":
    # 在本机5000端口运行
    app.run("127.0.0.1", port=5000)
