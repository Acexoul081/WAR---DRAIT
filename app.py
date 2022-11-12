from flask import Flask, render_template, request
import yaml
import json
import plotly
import plotly.express as px

app = Flask(__name__)

@app.route('/')
def main_page():
    with open("preprocess.yaml") as f:
        config = yaml.safe_load(f)
    model_data = []
    if config['models'] is not None:
        for model in config['models']:
            model_data.append({'key':model['key'], 'id':model['id']})
    return render_template('main.html', model_data=model_data)

@app.route('/metric/<metric>')
def show_metric(metric):
    return f'Hello {metric} !'

@app.route('/about-us')
def show_aboutus():
    return f''