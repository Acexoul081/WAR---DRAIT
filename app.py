from flask import Flask, render_template, request, session
import yaml
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pandas as pd
import db

app = Flask(__name__)
app.secret_key = 'any random string'

first_index= 1
last_index = 200
dataset = pd.read_csv('server_monitoring_log.csv', parse_dates=['date'], names=['date', 'cpu', 'memory', 'disk'])
dataset = dataset.set_index('date')
del dataset['memory']
del dataset['disk']

@app.route('/')
def main_page():
    with open("preprocess.yaml") as f:
        config = yaml.safe_load(f)
    model_data = []
    if config['models'] is not None:
        for model in config['models']:
            model_data.append({'key':model['key'], 'id':model['id']})
            model_data.append({'key':model['key'], 'id':model['id']})
            model_data.append({'key':model['key'], 'id':model['id']})
            model_data.append({'key':model['key'], 'id':model['id']})
    return render_template('main.html', model_data=model_data)

@app.route('/metric/<metric>', methods=['GET', 'POST'])
def show_metric(metric):
    global first_index, last_index, dataset
    if request.method == 'GET':
        #dummy
        temp = dataset[first_index:last_index]
        threshold = get_static_threshold()
        session['threshold'] = 'static'

        anomalies = get_anomalies(metric, temp)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=temp.index, y=temp['cpu'], name='Time Series'))
        fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies['cpu'], mode='markers', name='Anomaly'))
        fig.update_layout(showlegend=True, title='Detected anomalies')
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        return render_template('metric.html', metric=metric, graphJSON=graphJSON)

    if request.method == 'POST':
        # metric = request.args.get('data')
        # Session = sessionmaker(bind=db.engine)
        # session = Session()

        # result = session.query(db.anomalies).all()
    
        first_index+=1
        last_index+=1
        temp = dataset[first_index:last_index]

        anomalies = get_anomalies(metric, temp)

        # fig = go.Figure()
        # fig.add_trace(go.Scatter(x=test_score_df.index, y=test_score_df['value'], name='Time Series'))
        # fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies['value'], mode='markers', name='Anomaly'))
        # fig.add_trace(go.Scatter(x=temp.index, y=temp['cpu'], name='Time Series'))
        # fig.update_layout(showlegend=True, title='Detected anomalies')
        # graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        temp_datetime = [i.to_pydatetime() for i in temp.index]
        temp_datetime_anom = [i.to_pydatetime() for i in anomalies.index]
        return {
            'x':json.dumps(temp_datetime, default=str),
            'y': json.dumps(temp['cpu'].values.tolist()), 
            'x_anom': json.dumps(temp_datetime_anom, default=str), 
            'y_anom':json.dumps(anomalies['cpu'].values.tolist())
        }

def get_anomalies(metric, dataset):
    if session['threshold'] == 'static':
        threshold = get_static_threshold()
        dataset['anomaly'] = dataset['cpu'].ge(threshold)
    elif session['threshold'] == 'dynamic':
        threshold = get_dynamic_threshold()
        dataset['anomaly']=False
        print(threshold)
        for start_index, end_index in threshold:
            #validasi index outofbound
            print(start_index)
            print(end_index)
            dataset['anomaly'][start_index: end_index] = True

    anomalies = dataset.loc[dataset['anomaly'] == True]
    return anomalies

def get_models_version(metric):
      print(metric)

@app.route('/static', methods=['POST'])
def change_static_threshold():
    metric = request.args.get('data')
    session['threshold'] = 'static'
    return '', 204
    #query threshold data from db
    #create figure in json with anomaly by static threshold
    # test_score_df = pd.DataFrame(df_test[TIME_STEPS:])
    # test_score_df['loss'] = test_mae_loss
    # test_score_df['threshold'] = threshold
    # test_score_df['anomaly'] = test_score_df['loss'] > test_score_df['threshold']
    # test_score_df['Close'] = df_test[TIME_STEPS:]['value']

    # anomalies = test_score_df.loc[test_score_df['anomaly'] == True]

def get_static_threshold():
    static_threshold = 50
    return static_threshold

@app.route('/dynamic', methods=['POST'])
def change_dynamic_threshold():
    metric = request.args.get('data')
    session['threshold'] = 'dynamic'
    return '', 204
    # #query threshold data from db
    # #create figure in json with anomaly by dynamic threshold
    # test_score_df = pd.DataFrame(df_test[TIME_STEPS:])
    # test_score_df['loss'] = test_mae_loss
    # test_score_df['anomaly'] = test_score_df['loss'] > test_score_df['threshold']
    # test_score_df['Close'] = df_test[TIME_STEPS:]['value']

    # anomalies = test_score_df.loc[test_score_df['anomaly'] == True]

def get_dynamic_threshold():
    dynamic_threshold = [('2022-11-01 21:15:01', '2022-11-01 21:20:01')]
    return dynamic_threshold

@app.route('/about-us')
def show_aboutus():
    return f''