from flask import request, render_template, Flask
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
with open('coluas.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route("/")
def f():
    return render_template("index.html")

@app.route("/inspect")
def predict():  
    return render_template("inspect.html")
    
@app.route("/output", methods=["POST", "GET"])  # route to show the predictions in a web UI
def submit():
    
    input_feature = [float(x) if x.replace('.', '', 1).isdigit() else 0.0 for x in request.form.values()]
    x=[np.array(input_feature)]
    names = ["state","isMetro","areaname","county","family_member_count","housing_cost","food_cost","transportation_cost","healthcare_cost","other_necessities","childcare_cost","taxes","median_family_income"]
    data = pd.DataFrame(x,columns=names)
    predict = model.predict(data)
    
    return render_template('output.html', predict = predict)
    
if __name__ == "__main__":
    app.run(debug=True,port = 2222)