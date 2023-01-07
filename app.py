from flask import Flask, render_template, request, session, jsonify
import yaml
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, asc, and_, func, text
from sqlalchemy.orm import sessionmaker
from paramiko import SSHClient, AutoAddPolicy
import base64
import pandas as pd
import numpy as np

import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow_serving.apis import model_service_pb2_grpc
from tensorflow_serving.apis import get_model_status_pb2
from tensorflow_serving.apis import get_model_metadata_pb2
from google.protobuf.json_format import MessageToJson
from cron_descriptor import get_description, ExpressionDescriptor

import db

app = Flask(__name__)
app.secret_key = 'any random string'

client = SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(AutoAddPolicy())
client.connect('172.16.11.137', username='admin', password='P@ssword1234')

infra_metadata = {}
transaction_metadata = {}
with open("preprocess.yaml") as f:
    config = yaml.safe_load(f)
    for model in config['models']:
        infra_metadata[model['id']] = {
            'name': model['name'], 
            'tag': model['tag'],
            'key': model['key']
        }

with open("realtime transaction.yaml") as f:
    config = yaml.safe_load(f)
    for model in config['models']:
        transaction_metadata[model['id']] = {
            'key': model['key']
        }

#setting connection to tf-serving
PORT = 8500
channel = grpc.insecure_channel('172.16.11.137:{}'.format(PORT))
Session = sessionmaker(bind=db.engine)

@app.route('/')
def main_page():
    global config
    infra_data = []
    trans_data = []
    if infra_metadata:
        for id, model in infra_metadata.items():
            key_split = model['key'].split('|')
            ip = key_split[0]
            data_source = key_split[1]
            metric_type = key_split[3]
            infra_data.append({
                'key':model['key'],
                'id':id,
                'ip':ip,
                'source':data_source,
                'metric_type':metric_type
            })
    if transaction_metadata:
        for id, model in transaction_metadata.items():
            key_split = model['key'].split('|')
            print(key_split)
            app_name = key_split[1]
            data_source = key_split[0]
            metric_type = key_split[7]
            trans_data.append({
                'key':model['key'],
                'id':id,
                'app_name':app_name,
                'source':data_source,
                'metric_type':metric_type,
            })
    return render_template('main.html', infra_data=infra_data, trans_data=trans_data)

@app.route('/metric/<metric>', methods=['GET', 'POST'])
def show_metric(metric):
    global first_index, last_index
    if request.method == 'GET':
        session['threshold'] = 'static'
        decoded_metric = base64.urlsafe_b64decode(metric).decode("ascii")
        value, loss = get_value(metric)

        anomalies, threshold = get_anomalies(metric, loss)
        value_anomalies = value[value['metric_datetime'].isin(anomalies)]
        loss_anomalies = loss[loss['metric_datetime'].isin(anomalies)]

        model_version = get_model_version(metric)

        stdin,stdout,stderr = client.exec_command(f"crontab -l | grep -F \"{decoded_metric}\"")
        cron_info = stdout.read().decode('utf-8').strip().split('\n')
        cron_list = []

        if len(cron_info) > 0:
            print(cron_info)
            for i in cron_info:
                if len(i) > 0:
                    cron_list.append({
                        'job_detail':i,
                        'schedule':i[0:i.index("p")],
                        'schedule_readable':get_description(i[0:i.index("p")])
                    })

        val_graph_json = create_value_graph(value, value_anomalies, 'metric_value')
        loss_graph_json = create_value_graph(loss, loss_anomalies, 'loss', threshold)
        preproc_graph_json = create_value_graph(loss, loss_anomalies, 'value')
        
        decoded_metric_in_list = decoded_metric.split('|')
        metric_detail = {
            'ip':decoded_metric_in_list[0],
            'source':decoded_metric_in_list[1],
            'name':decoded_metric_in_list[2],
            'type':decoded_metric_in_list[3],
            'key': metric
        }
        
        return render_template('metric.html', metric=metric_detail, valueGraph=val_graph_json, lossGraph=loss_graph_json, preprocGraph= preproc_graph_json, modelStatus=model_version, crons = cron_list if cron_info else 'No Cron Available')

    elif request.method == 'POST':
        return get_value_json(metric)

def get_value_json(metric):
    value, loss = get_value(metric)
    anomalies, threshold = get_anomalies(metric, loss)
    value_anomalies = value[value['metric_datetime'].isin(anomalies)]
    loss_anomalies = loss[loss['metric_datetime'].isin(anomalies)]

    value_datetime = [i.to_pydatetime() for i in value['metric_datetime']]
    loss_datetime = [i.to_pydatetime() for i in loss['metric_datetime']]

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

def create_value_graph(dataset, anomalies, target_column, threshold=None):
    dataset = dataset.set_index('metric_datetime')
    dataset = dataset.resample('1T').mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dataset.index, y=dataset[target_column], name='Value Time Series'))
    fig.add_trace(go.Scatter(x=anomalies['metric_datetime'], y=anomalies[target_column], mode='markers', name='Anomaly'))
    if threshold:
        if session['threshold'] == 'static':
            fig.add_hline(y=threshold)
        else:
            for start_idx, end_idx, limit in threshold:
                fig.add_shape(type='line',
                    x0=dataset.index[start_idx],
                    y0=limit,
                    x1=dataset.index[end_idx],
                    y1=limit,
                    line=dict(color='Red',),
                    xref='x',
                    yref='y'
                )
    fig.update_layout(showlegend=True, title=target_column)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

def get_value(metric):
    db_session = Session()
    if metric in infra_metadata:
        loss_metric_key = infra_metadata[metric]['key']
    else:
        loss_metric_key = transaction_metadata[metric]['key']

    loss = db_session.query(db.losses).filter(
    and_(
        db.losses.metric_key == loss_metric_key,
        func.TIMESTAMPDIFF(text('month'),db.losses.timestamp,func.now()) <= 1
    )
    ).order_by(asc(db.losses.timestamp))

    loss = pd.read_sql(loss.statement, loss.session.bind)
    start_datetime = loss.iloc[0, loss.columns.get_loc('timestamp')].to_pydatetime()
    if metric in infra_metadata:
        value = db_session.query(db.metrics).filter(
        and_(
            db.metrics.tag == infra_metadata[metric]['tag'], 
            db.metrics.metric_name == infra_metadata[metric]['name']),
            func.TIMESTAMPDIFF(text('month'),db.metrics.metric_datetime,func.now()) <= 1,
            db.metrics.metric_datetime > start_datetime
        ).order_by(asc(db.metrics.metric_datetime))
    else:
        value = db_session.query(db.transactions).filter(
        and_(
            db.transactions.tag == transaction_metadata[metric]['key'], 
            func.TIMESTAMPDIFF(text('month'),db.transactions.metric_datetime,func.now()) <= 1,
            db.transactions.metric_datetime > start_datetime
        )
        ).order_by(asc(db.transactions.metric_datetime))
        
    value = pd.read_sql(value.statement, value.session.bind)
    loss.rename(columns = {'timestamp':'metric_datetime'}, inplace = True)
    return value,loss

def get_model_version(metric):
    global PORT, channel
    
    stub = model_service_pb2_grpc.ModelServiceStub(channel)
    request = get_model_status_pb2.GetModelStatusRequest()
    request.model_spec.name = metric
    result = stub.GetModelStatus(request, 5)  # 5 secs timeout
    
    # stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    # request = get_model_metadata_pb2.GetModelMetadataRequest()
    # request.model_spec.name = metric
    # request.metadata_field.append("signature_def")
    # result = stub.GetModelMetadata(request, 5)  # 5 secs timeout

    result = json.loads(MessageToJson(result))

    return result['model_version_status']

def get_anomalies(metric, dataset):
    dataset['anomaly']=False
    threshold = None
    if session['threshold'] == 'static':
        threshold = get_static_threshold(metric)
        if threshold is not None:
            dataset['anomaly'] = dataset['loss'].ge(threshold)
    elif session['threshold'] == 'dynamic':
        anom = get_dynamic_threshold(metric, dataset.iloc[-1]['metric_datetime'].to_pydatetime(),dataset['loss'])
        
        for start_index, end_index in zip(anom['start_time'],anom['end_time']):
            mask = (dataset['metric_datetime'] > start_index) & (dataset['metric_datetime'] <= end_index)
            dataset.loc[mask,'anomaly'] = True

    anomalies = dataset.loc[dataset['anomaly'] == True]['metric_datetime']
    return anomalies, threshold

@app.route('/static', methods=['POST'])
def change_static_threshold():
    metric = request.form['metric']
    session['threshold'] = 'static'
    return get_value_json(metric)

def get_static_threshold(metric):
    db_session = Session()
    if metric in infra_metadata:
        threshold_metric_key = infra_metadata[metric]['key']
    else:
        threshold_metric_key = transaction_metadata[metric]['key']
    threshold = db_session.query(db.thresholds).filter(db.thresholds.metric_key == threshold_metric_key)
    threshold = pd.read_sql(threshold.statement, threshold.session.bind)

    if threshold.empty:
        return None

    return threshold['static_threshold'].values[0]

@app.route('/dynamic', methods=['POST'])
def change_dynamic_threshold():
    metric = request.form['metric']
    session['threshold'] = 'dynamic'
    return get_value_json(metric)

def get_dynamic_threshold(metric, first_date, loss):
    db_session = Session()
    if metric in infra_metadata:
        anomaly_metric_key = infra_metadata[metric]['key']
    else:
        anomaly_metric_key = transaction_metadata[metric]['key']
        
    threshold_query = db_session.query(db.anomalies).filter(
        and_(
            db.anomalies.metric_key == anomaly_metric_key,
            db.anomalies.end_time > first_date
        )   
        ).order_by(asc(db.anomalies.start_time))
    dynamic_threshold = pd.read_sql(threshold_query.statement, threshold_query.session.bind)

    return dynamic_threshold

@app.route('/update-cron', methods=['POST'])
def update_cron_tab():
    new_cron = request.form['new-cron']
    prev_cron = request.form['prev-cron']
    new_cron_script = prev_cron.replace(prev_cron[0 : len(new_cron)], new_cron, 1)
    
    stdin,stdout,stderr = client.exec_command(f"crontab -l | grep -v '{prev_cron[len(new_cron)+1:].split('$')[0]}'  | crontab -")
    stdin,stdout,stderr = client.exec_command(f"(crontab -l ; echo '{new_cron_script}') | crontab -")
    return '', 204


@app.route('/about-us')
def show_aboutus():
    return render_template('about-us.html')

