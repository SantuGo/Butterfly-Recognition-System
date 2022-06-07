# 设置开发环境
ENV = 'development'
DEBUG = True

# 连接数据库
HOSTNAME = "127.0.0.1"
PORT = "3306"
DATABASE = "graduation_project"
USERNAME = "root"
PASSWORD = "root"
DB_URI = 'mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8'.format(USERNAME, PASSWORD, HOSTNAME, PORT, DATABASE)

SECRET_KEY = '123456'
SQLALCHEMY_DATABASE_URI = DB_URI

SQLALCHEMY_TRACK_MODIFICATIONS = True

ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']
