from sqlalchemy import create_engine
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base

engine = create_engine('mariadb+mariadbconnector://dbadmin:dbadmin@172.16.11.113/ai_db?charset=utf8mb4', echo=True)
#manage tables
base = declarative_base()

class anomalies(base):
    __tablename__='udd_anomalies'
    timeseries_id = Column(String(255), primary_key=True)
    timestamp = Column(DateTime)
    loss = Column(Float)
    is_anomaly = Column(Boolean, nullable=True)
    created_by = Column(String(50), primary_key=True)
    created_time = Column(DateTime)
    updated_by = Column(String(50), primary_key=True)
    updated_time = Column(DateTime)

    def __init__(self, timeseries_id, timestamp, loss, is_anomaly, created_by, created_time, updated_by, updated_time):
        self.timeseries_id = timeseries_id
        self.timestamp = timestamp
        self.loss = loss
        self.is_anomaly = is_anomaly
        self.created_by = created_by
        self.created_time = created_time
        self.updated_by = updated_by
        self.updated_time = updated_time