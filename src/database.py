from sqlalchemy import (
    create_engine, Column, Integer,
    Float, String, DateTime, Boolean
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

# Database URL
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://user:password@localhost:5432/maintenance_db"
)

Base = declarative_base()

class SensorReading(Base):
    __tablename__ = "sensor_readings"

    id = Column(Integer, primary_key=True)
    machine_id = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    temperature = Column(Float)
    vibration = Column(Float)
    pressure = Column(Float)
    rpm = Column(Float)
    voltage = Column(Float)
    current = Column(Float)
    predicted_failure = Column(Boolean)
    failure_probability = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class Database:
    def __init__(self):
        self.engine = create_engine(DATABASE_URL)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def save_prediction(self, data, prediction,
                        probability):
        reading = SensorReading(
            machine_id=data['machine_id'],
            temperature=data['temperature'],
            vibration=data['vibration'],
            pressure=data['pressure'],
            rpm=data['rpm'],
            voltage=data['voltage'],
            current=data['current'],
            predicted_failure=bool(prediction),
            failure_probability=float(probability)
        )
        self.session.add(reading)
        self.session.commit()
        return reading.id

    def get_recent_readings(self, machine_id,
                            limit=10):
        return (
            self.session.query(SensorReading)
            .filter_by(machine_id=machine_id)
            .order_by(SensorReading.timestamp.desc())
            .limit(limit)
            .all()
        )