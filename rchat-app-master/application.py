import os
import time
import webbrowser

from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, login_user, current_user, logout_user
from flask_socketio import SocketIO, join_room, leave_room, send

from wtform_fields import *
from models import *
import chatbot
import textwrap

#counter
num_client = 0

# Configure app
app = Flask(__name__)
app.secret_key= b'\xdej\xde\x84_a\xceE\x81\xf1\x88\x04\xafG\xb7\xb1'
app.config['WTF_CSRF_SECRET_KEY'] = "b'f\xfa\x8b{X\x8b\x9eM\x83l\x19\xad\x84\x08\xaa"

# Configure database
app.config['SQLALCHEMY_DATABASE_URI'] = "your_heroku_key"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Initialize login manager
login = LoginManager(app)
login.init_app(app)

@login.user_loader
def load_user(id):
    return User.query.get(int(id))

socketio = SocketIO(app, manage_session=False)

# Predefined rooms for chat
ROOMS = ["Chatbot", "Agent"]

@app.route("/show_history", methods=["GET","POST"])
def show_history():

    chat_history = Chat_history()
    if request.method == 'GET':
        return render_template("/show_history.html", query=chat_history.query.all())

    return redirect(url_for('show_history'), query=chat_history.query.all())


@app.route("/closepage",methods=['GET', 'POST'])
def closepage():
    return render_template("closepage.html")

@app.route("/review", methods=['GET', 'POST'])
def review():

    rev_form = ReviewForm()

    # Update database if validation success
    if rev_form.validate_on_submit():
        username = current_user.username
        rating = rev_form.rating.data
        comment = rev_form.comment.data
        # Add username & hashed password to DB
        review_form = Review(username=username, rating=rating, comment=comment)
        db.session.add(review_form)
        db.session.commit()

        flash('Submit successfully.', 'success')
        return redirect(url_for('closepage'))

    return render_template("review.html", form=rev_form)

@app.route("/submit_complaint", methods=['GET', 'POST'])
def submit_complaint():

    com_form = ComplaintForm()

    # Update database if validation success
    if com_form.validate_on_submit():
        username = current_user.username
        complaint = com_form.complaint.data
        # Add username & hashed password to DB
        complaint = Complaint(username=username, complaint=complaint)
        db.session.add(complaint)
        db.session.commit()

        flash('Submit successfully.', 'success')
        return redirect(url_for('closepage'))

    return render_template("complaint.html", form=com_form)

@app.route("/register", methods=['GET','POST'])
def register():

    reg_form = RegistrationForm()

    # Update database if validation success
    if reg_form.validate_on_submit():
        username = reg_form.username.data
        password = reg_form.password.data
        position = reg_form.position.data
        

        # Hash password
        hashed_pswd = pbkdf2_sha256.hash(password)

        # Add username & hashed password to DB
        user = User(username=username, hashed_pswd=hashed_pswd, position=position)
        db.session.add(user)
        db.session.commit()

        flash('Registered successfully!', 'success')
        return redirect(url_for('admin_chat'))

    return render_template("register.html", form=reg_form)

@app.route("/", methods=['GET', 'POST'])
def index():
    return redirect(url_for("login"))


@app.route("/visitor", methods=['GET','POST'])
def visitor():

    reg_form = VisitorRegistrationForm()

    # Update database if validation success
    if reg_form.validate_on_submit():
        username = reg_form.username.data
        password = reg_form.password.data
        position = reg_form.position.data

        # Hash password
        hashed_pswd = pbkdf2_sha256.hash(password)

        # Add username & hashed password to DB
        user = User(username=username, hashed_pswd=hashed_pswd, position=position)
        db.session.add(user)
        db.session.commit()

        flash('Registered successfully. Please login.', 'success')
        return redirect(url_for('login'))

    return render_template("visitor.html", form=reg_form)

@app.route("/login", methods=['GET', 'POST'])
def login():

    login_form = LoginForm()

    # Allow login if validation success
    if login_form.validate_on_submit():
        user_object = User.query.filter_by(username=login_form.username.data, position=login_form.position.data).first()
        if login_user(user_object) and login_form.position.data == "Admin":
            return redirect(url_for("admin_chat"))
        else:
            return redirect(url_for('chat'))

    return render_template("login.html", form=login_form)



@app.route("/logout", methods=['GET'])
def logout():

    # Logout user
    logout_user()
    flash('You have logged out successfully', 'success')
    return redirect(url_for('login'))

@app.route("/admin_chat", methods=['GET', 'POST'])
def admin_chat():

    if not current_user.is_authenticated:
        flash('Please login', 'danger')
        return redirect(url_for('login'))

    return render_template("admin_chat.html", username=current_user.username, rooms=ROOMS)

@app.route("/chat", methods=['GET', 'POST'])
def chat():

    if not current_user.is_authenticated:
        flash('Please login', 'danger')
        return redirect(url_for('login'))

    return render_template("chat.html", username=current_user.username, rooms=ROOMS)


@app.errorhandler(404)
def page_not_found(e):
    # note that we set the 404 status explicitly
    return render_template('404.html'), 404


@socketio.on('incoming-msg')
def on_message(data):
    """Broadcast messages"""

    msg = data["msg"]
    username = data["username"]
    room = data["room"]

    # Set timestamp
    time_stamp = time.strftime('%b-%d %I:%M%p', time.localtime())
    if num_client == 1:

        send({"username": username, "msg": msg, "time_stamp": time_stamp}, room=room)
        chat_history = Chat_history(username=username, history=msg)
        db.session.add(chat_history)
        db.session.commit()
        tag = chatbot.get_bot_response(msg)
        print(tag)
        if tag == "cancel_order":
            response = """Yes, you will be able to cancel your order using the steps below if:
                You have not made full payment to your order
                The seller has not arranged shipment (The order does not have any tracking status updated yet).
                You have not requested for cancellation for this order before this. (You are allowed to request for cancellation once only per order. The order will resume shipping process if you have previously withdrawn your cancellation request / cancellation request rejected by seller. Don't worry, order will be automatically cancelled by the system if seller does not ship 3 days after Days to Ship.
                """
            send({"username": 'Bot', "msg": response, "time_stamp": time_stamp}, room=room)

        elif tag == "unknown":
            response = "I didn't quite get that, please try again."
            send({"username": 'Bot', "msg": response, "time_stamp": time_stamp}, room=room)

        elif tag == "contact_customer_service":
            response = "Got it, contacting an agent to assist you, please wait for a while"
            send({"username": 'Bot', "msg": response, "time_stamp": time_stamp}, room=room)

            webbrowser.open_new('http://127.0.0.1:8000/')
        elif tag == "check_refund_policy":
            response = """Buyer will only be refunded after Shopee has received the confirmation from Seller that Seller has received the returned Item. In the event where Shopee does not hear from Seller within a specified time, Shopee will be at liberty to refund the applicable sum to Buyer without further notice to Seller. For more information on Seller’s response time limits, please click this link. The refund will be made to Buyer’s credit/debit card or designated bank account, whichever is applicable."""
            send({"username": 'Bot', "msg": response, "time_stamp": time_stamp}, room=room)

        elif tag == "delivery_options":
            response = '''Buyers will be able to view and choose from one of the following shipping options that suit their needs under Standard Service Types. 
                        \n•	Standard Delivery
                        \n•	Economy Delivery (Sea Shipping)
                        \n•	Others (West Malaysia/East Malaysia)*
                        
                        \n*Note: Shipping channels under Others (West Malaysia/East Malaysia) will fall under Non-Shopee Supported Logistics. 
                        
                        \nShopee will then allocate the order to the most suitable logistics provider for delivery. Buyers will be able to view the allocated Logistic Provider in the Order Details Page once: 
                        
                        \nBefore 23 June 2021
                        \n•	You have paid for your order.
                        \n•	The system is done allocating the most suitable Logistic Provider.
            '''
            send({"username": 'Bot', "msg": response, "time_stamp": time_stamp}, room=room)

        elif tag == "complaint":
            response = "Ok, Bot is redirecting to complaint page"
            send({"username": 'Bot', "msg": response, "time_stamp": time_stamp}, room=room)

            webbrowser.open_new('http://127.0.0.1:5000/submit_complaint')
        elif tag == "delivery_period":
            response = "Delivery period is vary based on delivery period. Postal Shipping takes 3 to 4 weeks, PosLaju takes 1 weeks."
            send({"username": 'Bot', "msg": response, "time_stamp": time_stamp}, room=room)

        elif tag == "delete_account":
            response = "You can delete account by sending request to the administrator."
            send({"username": 'Bot', "msg": response, "time_stamp": time_stamp}, room=room)

        elif tag == "edit_account":
            response = "Showing how to edit account"
            send({"username": 'Bot', "msg": response, "time_stamp": time_stamp}, room=room)

        elif tag == "get_refund":
            response = """When can I apply for a Return / Refund ?   
                        If your order is still under "To Receive" tab, please proceed to request for return/refund if you:
                        Did not receive the order 
                        Received an incomplete product (missing quantity or accessories)
                        Received the wrong product(s) (e.g. wrong size, wrong colour, different product)
                        Received a product with physical damage (e.g. dented, scratched, broken) 
                        Received a faulty product (e.g. malfunction, does not work as intended)"""
            send({"username": 'Bot', "msg": response, "time_stamp": time_stamp}, room=room)

        elif tag == "recover_password":
            response = """Make sure that you have verified your email address and mobile number before resetting the password to be able to receive verification code via SMS or Email.
                        Follow these steps on how to reset your password:
                        Step 1: Tap the “Forgot?” in Log in.
                        Step 2: Enter your email address or mobile number and tap “Next”.
                        Step 3: You will then be prompted to enter the code.
                        Step 4: You will receive a text or email with the verification code. Enter the verification code, tap Next or tap the link within the email.
                        Step 5: Enter your new password, and tap “Reset”.
                        After you reset your password you will now be able to log in with your new password. If you have other concerns logging in, please refer here."""
            send({"username": 'Bot', "msg": response, "time_stamp": time_stamp}, room=room)

        elif tag == "review":
            response = "Got it, redirecting to review form"
            send({"username": 'Bot', "msg": response, "time_stamp": time_stamp}, room=room)

            webbrowser.open_new('http://127.0.0.1:5000/review')
    elif num_client > 1:
        send({"username": username, "msg": msg, "time_stamp": time_stamp}, room=room)
        chat_history = Chat_history(username=username, history=msg)
        db.session.add(chat_history)
        db.session.commit()

@socketio.on('join')
def on_join(data):
    """User joins a room"""
    global num_client
    if num_client >= 3:
        flash("There are already someone in chat room, Please join other room")
    else:
        username = data["username"]
        room = data["room"]
        join_room(room)
        num_client = num_client + 1

    # Broadcast that new user has joined
    send({"msg": username + " has joined the " + room + " room."}, room=room)
    if num_client == 1:
        send({"username": 'Bot',"msg": 'Hi, I am AlexBot, a customer service bot that assist customer, AlphaBot can answer the question that is related to cancel order, check refund policy, complaint, contact customer service, contact human agent, delete account, delivery options, delivery period, edit account, get refund, recover password and review. What can I do for you today?'}, room=room)


@socketio.on('leave')
def on_leave(data):
    """User leaves a room"""
    global num_client
    username = data['username']
    room = data['room']

    if num_client > 0:
        leave_room(room)
        num_client = num_client - 1
    print(num_client)

    send({"msg": username + " has left the room"}, room=room)

if __name__ == "__main__":
    socketio.run(app)
