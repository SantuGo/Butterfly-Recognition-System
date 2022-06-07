from exts import db
from datetime import datetime


class AdminModel(db.Model):
    __tablename__ = "Admin"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    admin_name = db.Column(db.String(200), nullable=False, unique=True)
    email = db.Column(db.String(100), nullable=True, unique=True)
    password = db.Column(db.String(200), nullable=False)
