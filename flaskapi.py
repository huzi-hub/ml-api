from flask import Flask, request
import pickle
import socket
import financialanalysis as fa

app = Flask(__name__)

@app.route('/predict', methods = ['GET'])
def predict():
    with open('lassoModel.pkl', 'rb') as f:
        model1 = pickle.load(f)
    with open('linearModel.pkl', 'rb') as f:
        model2 = pickle.load(f)
    with open('ridgeModel.pkl', 'rb') as f:
        model3 = pickle.load(f)
    y=list(request.args.to_dict().values())[0]
    y=fa.datetimeToFloatyear(fa.stringToDatetime(y))
    ypred1=int(model1.predict([[y]])[0])
    ypred2=int(model2.predict([[y]])[0])
    ypred3=int(model3.predict([[y]])[0])
    if ypred1 > 0 and ypred2 > 0 and ypred3 > 0:
        out="Input => "+str(y)+" LASSO MODEL PREDICTION => "+ str(ypred1) +"\n LINEAR MODEL PREDICTION => "+ str(ypred2) +"\n RIDGE MODEL PREDICTION => "+ str(ypred3)    
        return out
    elif ypred1 < 0 and ypred2 < 0 and ypred3 < 0: 
        out="Input => "+str(y)+" Output => enter the corrent date"
        return out
    
@app.route('/')
def index():
    return "HELLO"
 
if __name__ =="__main__":
    addr = socket.gethostbyname(socket.gethostname())
    app.run(addr,5000,debug=True)