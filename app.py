from flask import Flask, render_template, request, session, jsonify
import yaml
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, desc, and_
from sqlalchemy.orm import sessionmaker

import pandas as pd

import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow_serving.apis import model_service_pb2_grpc
from tensorflow_serving.apis import get_model_status_pb2
from tensorflow_serving.apis import get_model_metadata_pb2
from google.protobuf.json_format import MessageToJson

import db

#setting app backend
app = Flask(__name__)
app.secret_key = 'any random string'

model_metadata = {}
with open("preprocess.yaml") as f:
    config = yaml.safe_load(f)
    for model in config['models']:
        model_metadata[model['id']] = {
            'name': model['name'], 
            'tag': model['tag'],
            'key': model['key']
        }

#setting connection to tf-serving
PORT = 8500
channel = grpc.insecure_channel('localhost:{}'.format(PORT))
Session = sessionmaker(bind=db.engine)

@app.route('/')
def main_page():
    global config
    model_data = []
    if config['models'] is not None:
        for model in config['models']:
            model_data.append({'key':model['key'], 'id':model['id']})
    return render_template('main.html', model_data=model_data)

@app.route('/metric/<metric>', methods=['GET', 'POST'])
def show_metric(metric):
    global first_index, last_index
    if request.method == 'GET':
        value, loss = get_value(metric)
        session['threshold_value'] = get_static_threshold(metric)

        anomalies = get_anomalies(metric, loss)
        value_anomalies = value.iloc[anomalies]
        loss_anomalies = loss.iloc[anomalies]
        # model_versions = get_model_version(metric)
        
        val_graph_json = create_value_graph(value, value_anomalies, 'metric_value')
        loss_graph_json = create_value_graph(loss, loss_anomalies, 'loss')

        return render_template('metric.html', metric=metric, valueGraph=val_graph_json, lossGraph=loss_graph_json)

    elif request.method == 'POST':   
        value, loss = get_value(metric)
        anomalies = get_anomalies(metric, loss)
        value_anomalies = value.iloc[anomalies]
        loss_anomalies = loss.iloc[anomalies]

        print(value_anomalies)
        print(loss_anomalies)

        value_datetime = [i.to_pydatetime() for i in value['metric_datetime']]
        loss_datetime = [i.to_pydatetime() for i in loss['metric_datetime']]
        #ganti universal anomalies ntar
        value_anom_datetime = [i.to_pydatetime() for i in value_anomalies['metric_datetime']]
        loss_anom_datetime = [i.to_pydatetime() for i in loss_anomalies['metric_datetime']]
        return {
            'x_value':json.dumps(value_datetime, default=str),
            'y_value': json.dumps(value['metric_value'].values.tolist()), 
            'x_value_anom': json.dumps(value_anom_datetime, default=str), 
            'y_value_anom':json.dumps(value_anomalies['metric_value'].values.tolist()),
            'x_loss':json.dumps(loss_datetime, default=str),
            'y_loss': json.dumps(loss['loss'].values.tolist()), 
            'x_loss_anom': json.dumps(loss_anom_datetime, default=str), 
            'y_loss_anom':json.dumps(loss_anomalies['loss'].values.tolist())
        }

def create_value_graph(dataset, anomalies, target_column):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dataset['metric_datetime'], y=dataset[target_column], name='Value Time Series'))
    fig.add_trace(go.Scatter(x=anomalies['metric_datetime'], y=anomalies[target_column], mode='markers', name='Anomaly'))
    fig.update_layout(showlegend=True, title='Detected anomalies')
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

def get_value(metric):
    db_session = Session()
    loss = db_session.query(db.losses).filter(db.losses.metric_key == model_metadata[metric]['key']).order_by(desc(db.losses.timestamp)).limit(200)
    value = db_session.query(db.metrics).filter(
        and_(
            db.metrics.tag == model_metadata[metric]['tag'], 
            db.metrics.metric_name == model_metadata[metric]['name'])
        ).order_by(desc(db.metrics.metric_datetime)).limit(200)

    loss = pd.read_sql(loss.statement, loss.session.bind)
    value = pd.read_sql(value.statement, value.session.bind)
    loss.rename(columns = {'timestamp':'metric_datetime'}, inplace = True)
    return value,loss
    #buat yg metric ke transaction

def get_model_version(metric):
    global PORT, channel
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    request = get_model_status_pb2.GetModelStatusRequest()
    request.model_spec.name = metric
    result = stub.GetModelStatus(request, 5)  # 5 secs timeout
    print(f"Model status: {result}")
    
    request = get_model_metadata_pb2.GetModelMetadataRequest()
    request.model_spec.name = metric
    request.metadata_field.append("signature_def")
    result = stub.GetModelMetadata(request, 5)  # 5 secs timeout
    result = json.loads(MessageToJson(result))
    print(f"Model metadata: {result}")

def get_anomalies(metric, dataset):
    if session['threshold'] == 'static':
        threshold = get_static_threshold(metric)
        dataset['anomaly'] = dataset['loss'].ge(threshold)
    elif session['threshold'] == 'dynamic':
        threshold = get_dynamic_threshold()
        dataset['anomaly']=False
        print(threshold)
        for start_index, end_index in threshold:
            #validasi index outofbound
            dataset['anomaly'][start_index: end_index] = True
    
    #anomali samain antara value df sama loss df
    #ntar ambil anomali per datetime
    anomalies = dataset.loc[dataset['anomaly'] == True].index
    return anomalies

def get_models_version(metric):
    print(metric)

@app.route('/static', methods=['POST'])
def change_static_threshold():
    metric = request.args.get('data')
    session['threshold_value'] = get_static_threshold(metric)
    
    return '', 204

def get_static_threshold(metric):
    db_session = Session()
    session['threshold'] = 'static'
    threshold = db_session.query(db.thresholds).filter(db.thresholds.metric_key == model_metadata[metric]['key'])

    threshold = pd.read_sql(threshold.statement, threshold.session.bind)
    return threshold['static_threshold']

@app.route('/dynamic', methods=['POST'])
def change_dynamic_threshold():
    metric = request.args.get('data')
    session['threshold'] = 'dynamic'
    return '', 204
    # #query threshold data from db

def get_dynamic_threshold(metric):
    db_session = Session()
    threshold_query = db_session.query(db.anomalies).filter(db.anomalies.metric_key == model_metadata[metric]['key']).order_by(desc(db.anomalies.start_time)).limit(200) #ambil sampe jamber
    dynamic_threshold = pd.read_sql(threshold_query.statement, threshold_query.session.bind)
    return dynamic_threshold

@app.route('/about-us')
def show_aboutus():
    return render_template('about-us.html')