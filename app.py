from flask import Flask, render_template, request, session, jsonify
import yaml
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, desc, and_, func, text
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
        value_anomalies = value.iloc[anomalies]
        loss_anomalies = loss.iloc[anomalies]
        model_version = get_model_version(metric)

        stdin,stdout,stderr = client.exec_command(f"crontab -l | grep -F \"{decoded_metric}\"")
        cron_info = stdout.read().decode('utf-8').strip().split('\n')
        cron_list = []

        if len(cron_info) > 0:
            print(cron_info)
            for i in cron_info:
                if len(i) > 0:
                    job = ''
                    if 'train' in i:
                        job = "Train Model"
                    elif 'renew' in i:
                        job = "Update Preprocessing Data"
                    cron_list.append({
                        'job_detail':job,
                        'schedule':i[0:i.index("p")-1],
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
    value_anomalies = value.iloc[anomalies]
    loss_anomalies = loss.iloc[anomalies]

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
        value = db_session.query(db.metrics).filter(
        and_(
            db.metrics.tag == infra_metadata[metric]['tag'], 
            db.metrics.metric_name == infra_metadata[metric]['name']),
            func.TIMESTAMPDIFF(text('month'),db.metrics.metric_datetime,func.now()) <= 1
        ).order_by(desc(db.metrics.metric_datetime))
    else:
        loss_metric_key = transaction_metadata[metric]['key']
        value = db_session.query(db.transactions).filter(
        and_(
            db.transactions.tag == transaction_metadata[metric]['key'], 
            func.TIMESTAMPDIFF(text('month'),db.transactions.metric_datetime,func.now()) <= 1
        )
        ).order_by(desc(db.transactions.metric_datetime))
        

    loss = db_session.query(db.losses).filter(
        and_(
            db.losses.metric_key == loss_metric_key,
            func.TIMESTAMPDIFF(text('month'),db.losses.timestamp,func.now()) <= 1
        )
    ).order_by(desc(db.losses.timestamp))

    loss = pd.read_sql(loss.statement, loss.session.bind)
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
    if session['threshold'] == 'static':
        threshold = get_static_threshold(metric)
        if threshold is not None:
            dataset['anomaly'] = dataset['loss'].ge(threshold)
    elif session['threshold'] == 'dynamic':
        anom, threshold = get_dynamic_threshold(metric, dataset.iloc[-1]['metric_datetime'].to_pydatetime(),dataset['loss'])
        
        for start_index, end_index in zip(anom['start_time'],anom['end_time']):
            mask = (dataset['metric_datetime'] > start_index) & (dataset['metric_datetime'] <= end_index)
            dataset.loc[mask,'anomaly'] = True

    anomalies = dataset.loc[dataset['anomaly'] == True].index
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
        ).order_by(desc(db.anomalies.start_time))
    dynamic_threshold = pd.read_sql(threshold_query.statement, threshold_query.session.bind)

    dynamic_error = Errors(loss)
    epsilon = dynamic_error.process_batches()
    return dynamic_threshold, epsilon

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

import more_itertools as mit
import math

l_s = 60
batch_size = 32
window_size= 30
smoothing_perc= 0.05
error_buffer= 100
p= 0.13
class Errors:
    def __init__(self, e):
        self.window_size = window_size
        self.n_windows = int((e.shape[0] -
                              (batch_size * self.window_size))
                             / batch_size)
        self.i_anom = np.array([])
        self.E_seq = []
        self.E_point = []
        self.anom_scores = []
        self.train_from_begin = True

        self.e = e
        smoothing_window = int(batch_size * window_size
                               * smoothing_perc)

        # smoothed prediction error
        self.e_s = pd.DataFrame(self.e).ewm(span=smoothing_window)\
            .mean().values.flatten()

    def adjust_window_size(self, e):
        """
        Decrease the historical error window size (h) if number of test
        values is limited.

        Args:
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel
        """

        while self.n_windows < 0:
            self.window_size -= 1
            self.n_windows = int((e.shape[0]
                                 - (batch_size * self.window_size))
                                 / batch_size)
            if self.window_size == 1 and self.n_windows < 0:
                raise ValueError('Batch_size ({}) larger than y_test (len={}). '
                                 'Adjust in config.yaml.'
                                 .format(batch_size,
                                         e.shape[0]))

    def process_batches(self):
        """
        Top-level function for the Error class that loops through batches
        of values for a channel.

        Args:
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel
        """
        threshold_array = []
        self.adjust_window_size(self.e)
        for i in range(0, self.n_windows+1):
            prior_idx = i * batch_size
            idx = (window_size * batch_size) + (i * batch_size)
            if i == self.n_windows:
                idx = self.e.shape[0]
            
            window = ErrorWindow(prior_idx, idx, self, self.n_windows)

            window.find_epsilon()
            window.find_epsilon(inverse=True)
            threshold_array.append([prior_idx, idx,window.epsilon])
        return threshold_array   

class ErrorWindow:
    def __init__(self, start_idx, end_idx, errors, window_num):
        """
        Data and calculations for a specific window of prediction errors.
        Includes finding thresholds, pruning, and scoring anomalous sequences
        for errors and inverted errors (flipped around mean) - significant drops
        in values can also be anomalous.

        Args:
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel
            config (obj): Config object containing parameters for processing
            start_idx (int): Starting index for window within full set of
                channel test values
            end_idx (int): Ending index for window within full set of channel
                test values
            errors (arr): Errors class object
            window_num (int): Current window number within channel test values

        Attributes:
            i_anom (arr): indices of anomalies in window
            i_anom_inv (arr): indices of anomalies in window of inverted
                telemetry values
            E_seq (arr of tuples): array of (start, end) indices for each
                continuous anomaly sequence in window
            E_seq_inv (arr of tuples): array of (start, end) indices for each
                continuous anomaly sequence in window of inverted telemetry
                values
            non_anom_max (float): highest smoothed error value below epsilon
            non_anom_max_inv (float): highest smoothed error value below
                epsilon_inv
            config (obj): see Args
            anom_scores (arr): score indicating relative severity of each
                anomaly sequence in E_seq within a window
            window_num (int): see Args
            sd_lim (int): default number of standard deviations to use for
                threshold if no winner or too many anomalous ranges when scoring
                candidate thresholds
            sd_threshold (float): number of standard deviations for calculation
                of best anomaly threshold
            sd_threshold_inv (float): same as above for inverted channel values
            e_s (arr): exponentially-smoothed prediction errors in window
            e_s_inv (arr): inverted e_s
            sd_e_s (float): standard deviation of e_s
            mean_e_s (float): mean of e_s
            epsilon (float): threshold for e_s above which an error is
                considered anomalous
            epsilon_inv (float): threshold for inverted e_s above which an error
                is considered anomalous
            y_test (arr): Actual telemetry values for window
            sd_values (float): st dev of y_test
            perc_high (float): the 95th percentile of y_test values
            perc_low (float): the 5th percentile of y_test values
            inter_range (float): the range between perc_high - perc_low
            num_to_ignore (int): number of values to ignore initially when
                looking for anomalies
        """

        self.i_anom = np.array([])
        self.E_seq = np.array([])
        self.non_anom_max = -1000000
        self.i_anom_inv = np.array([])
        self.E_seq_inv = np.array([])
        self.non_anom_max_inv = -1000000

        self.anom_scores = []

        self.window_num = window_num

        self.sd_lim = 12.0
        self.sd_threshold = self.sd_lim
        self.sd_threshold_inv = self.sd_lim

        self.e_s = errors.e_s[start_idx:end_idx]

        self.mean_e_s = np.mean(self.e_s)
        self.sd_e_s = np.std(self.e_s)
        self.e_s_inv = np.array([self.mean_e_s + (self.mean_e_s - e)
                                 for e in self.e_s])

        self.epsilon = self.mean_e_s + self.sd_lim * self.sd_e_s
        self.epsilon_inv = self.mean_e_s + self.sd_lim * self.sd_e_s

        # ignore initial error values until enough history for processing
        self.num_to_ignore = l_s * 2


    def find_epsilon(self, inverse=False):
        """
        Find the anomaly threshold that maximizes function representing
        tradeoff between:
            a) number of anomalies and anomalous ranges
            b) the reduction in mean and st dev if anomalous points are removed
            from errors
        (see https://arxiv.org/pdf/1802.04431.pdf)

        Args:
            inverse (bool): If true, epsilon is calculated for inverted errors
        """
        e_s = self.e_s if not inverse else self.e_s_inv

        max_score = -10000000

        for z in np.arange(2.5, self.sd_lim, 0.5):
            epsilon = self.mean_e_s + (self.sd_e_s * z)

            pruned_e_s = e_s[e_s < epsilon] #ambil error yg aman

            i_anom = np.argwhere(e_s >= epsilon).reshape(-1,)
            buffer = np.arange(1, error_buffer)
            i_anom = np.sort(np.concatenate((i_anom,
                                            np.array([i+buffer for i in i_anom])
                                             .flatten(),
                                            np.array([i-buffer for i in i_anom])
                                             .flatten())))
            i_anom = i_anom[(i_anom < len(e_s)) & (i_anom >= 0)]
            i_anom = np.sort(np.unique(i_anom))

            if len(i_anom) > 0:
                # group anomalous indices into continuous sequences
                groups = [list(group) for group
                          in mit.consecutive_groups(i_anom)]
                E_seq = [(g[0], g[-1]) for g in groups if not g[0] == g[-1]]

                mean_perc_decrease = (self.mean_e_s - np.mean(pruned_e_s)) \
                                     / self.mean_e_s
                sd_perc_decrease = (self.sd_e_s - np.std(pruned_e_s)) \
                                   / self.sd_e_s
                score = (mean_perc_decrease + sd_perc_decrease) \
                        / (len(E_seq) ** 2 + len(i_anom))

                # sanity checks / guardrails
                if score >= max_score and len(E_seq) <= 5 and \
                        len(i_anom) < (len(e_s) * 0.5):
                    max_score = score
                    if not inverse:
                        self.sd_threshold = z
                        self.epsilon = self.mean_e_s + z * self.sd_e_s
                    else:
                        self.sd_threshold_inv = z
                        self.epsilon_inv = self.mean_e_s + z * self.sd_e_s

