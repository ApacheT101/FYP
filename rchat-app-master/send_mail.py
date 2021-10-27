from flask import Flask
from flask_mail import Mail, Message

app = Flask(__name__)

app.config.update(
    MAIL_SERVER = 'smtp.gmail.com',
    MAIL_PORT = 465,
    MAIL_USE_SSL = True,
    MAIL_USE_TLS = False,
    MAIL_USERNAME = 'fionleepeifong@gmail.com',
    MAIL_PASSWORD = 'weixian'
)

mail = Mail(app)

@app.route('/')
def index():
    msg = Message('Hello', sender='apachecompany00@gmail.com', recipients=['weixian1999@gmail.com'])
    msg.body = 'There is a customer need customer support, please sign in to assist the customer'
    mail.send(msg)
    return "Success";

if __name__ == "__main__":
    app.run(host="localhost", port=8000)