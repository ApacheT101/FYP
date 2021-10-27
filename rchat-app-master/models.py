from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

db = SQLAlchemy()


class User(UserMixin, db.Model):
    """ User model """

    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(25), unique=True, nullable=False)
    hashed_pswd = db.Column(db.String(), nullable=False)
    position = db.Column(db.String(), nullable=False)

class Complaint(db.Model):

    __tablename__ = "complaint"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(25), nullable=False)
    complaint = db.Column(db.String(), nullable=False)
    #user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

class Review(db.Model):

    __tablename__ = "review"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(25), nullable=False)
    rating = db.Column(db.Integer, nullable=False)
    comment = db.Column(db.String(), nullable=False)
    #user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

class Chat_history(db.Model):

    __tablename__ = "chat_history"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(25), nullable=False)
    history = db.Column(db.String(), nullable=False)