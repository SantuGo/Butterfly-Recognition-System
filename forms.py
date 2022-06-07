import wtforms
from wtforms.validators import length, email

class RegisterForm(wtforms.Form):
    admin_registername = wtforms.StringField(validators=[length(min=5, max=30)])
    register_password = wtforms.StringField(validators=[length(min=6, max=20)])

class LoginForm(wtforms.Form):
    admin_name = wtforms.StringField(validators=[length(min=5, max=30) or email()])
    password = wtforms.StringField(validators=[length(min=6, max=20)])
