from flask import Flask, render_template, request
import numpy as np
import function1
import config1

user = 'nitagutthe'
password ='12345'

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('log1.html')

@app.route('/login', methods = ['POST'])

def login():
    u1 = str(request.form.get('username'))
    p1 = str(request.form.get('password'))
    
    if user == u1 and password== p1:
        return render_template('index1.html')

    else:
        return render_template('log1.html')
    # return "pass"

@app.route('/predict',methods = ['POST'])
def loan_app():
    
    job = str(request.form.get('job'))
    marital =str(request.form.get('marital'))
    education = str(request.form.get('education'))
    default = str(request.form.get('default'))
    housing = str(request.form.get('housing'))
    loan = str(request.form.get('loan'))
    campaign = str(request.form.get('campaign'))
    previous = str(request.form.get('previous'))
    previous_outcome = str(request.form.get('previous_outcome'))
    

    
    print(job,marital,education,default,housing,loan,campaign,previous,previous_outcome)
    data = np.array([[job,marital,education,default,housing,loan,campaign,previous,previous_outcome]])
    result = function1.prediction(data)

    return render_template('index1.html',Offer_status_result=result)

if __name__ == "__main__":
    app.run(host=config1.HOST_NAME,port=config1.PORT_NUMBER)