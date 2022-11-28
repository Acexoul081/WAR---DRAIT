from curses import echo
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

engine = create_engine('mariadb+mariadbconnector://dbadmin:dbadmin@172.16.11.113/ai_db?charset=utf8mb4', echo=True)
#manage tables
base = declarative_base()

class anomalies(base):
    __tablename__='udd_anomalies'
    anomaly_id = Column(String(255), primary_key=True)
    metric_key = Column(String(255))
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    created_by = Column(String(255))
    created_time = Column(DateTime)
    updated_by = Column(String(255))
    updated_time = Column(DateTime)


    def __init__(self, anomaly_id, metric_key, start_time, end_time, created_by):
        self.anomaly_id = anomaly_id
        self.metric_key = metric_key
        self.start_time = start_time
        self.end_time = end_time
        self.created_by = created_by
        self.created_time = datetime.now()

class losses(base):
    __tablename__='udd_metrics2'
    timeseries_id = Column(String(255), primary_key=True)
    timestamp = Column(DateTime)
    loss = Column(Float)
    value = Column(Float)
    original = Column(Float)
    metric_key = Column(String(255))
    created_by = Column(String(255))
    created_time = Column(DateTime)
    updated_by = Column(String(255))
    updated_time = Column(DateTime)


    def __init__(self, timeseries_id, timestamp, loss,value,original, metric_key, created_by):
        self.timeseries_id = timeseries_id
        self.timestamp = timestamp
        self.loss = loss
        self.value = value
        self.original = original
        self.metric_key = metric_key
        self.created_by = created_by
        self.created_time = datetime.now()

class metrics(base):
    __tablename__='raw_metric_infrastructure'
    id = Column(String(255), primary_key=True)
    metric_datetime = Column(DateTime)
    metric_unixtime = Column(Float)
    data_source = Column(String(255))
    category = Column(String(255))
    application_name = Column(String(255))
    infrastructure_component = Column(String(255))
    location = Column(String(255))
    node = Column(String(255))
    node_instance = Column(String(255))
    ip = Column(String(255))
    metric_type = Column(String(255))
    metric_name = Column(String(255))
    metric_value = Column(Float)
    metric_unit = Column(String(255))
    tag = Column(String(255))
    metric_unit = Column(String(255))
    tag = Column(String(255))
    created_by = Column(String(255))
    created_time = Column(DateTime)
    updated_by = Column(String(255))
    updated_time = Column(DateTime)


    def __init__(self, timeseries_id, timestamp, loss,value,original, metric_key, created_by):
        self.timeseries_id = timeseries_id
        self.timestamp = timestamp
        self.loss = loss
        self.value = value
        self.original = original
        self.metric_key = metric_key
        self.created_by = created_by
        self.created_time = datetime.now()

# base.metadata.create_all(engine)