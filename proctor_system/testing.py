import numpy as np
from flask import Flask,request,jsonify,render_template,redirect,url_for
import cv2

app = Flask(__name__)

Name=""
@app.route('/')
def home():
    global Name
    Name="hi"
    print(Name)
    return redirect(url_for('quiz'))

@app.route('/quiz',methods=['POST','GET'])
def quiz():
    global Name
    print(Name)
    return render_template('quiz.html')

@app.route('/xyz')
def xyz():
    runn.delay()
    return render_template('work.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/run',methods=['POST','GET'])
def run():
    print("here")
    if request.method == 'POST':
        return redirect(url_for('checkk'))
    return render_template('quiz.html')

@app.route('/checkk',methods=['POST','GET'])
def checkk():
    return runn()

@app.route('/finish')
def finish(): 
    return render_template('Thank_you.html')

if __name__=="__main__":
    app.run(debug=True)
