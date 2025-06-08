from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Date, Float, BigInteger, Text
from sqlalchemy.ext.declarative import declarative_base
import os

load_dotenv()

Base = declarative_base()

class XHSData(Base):
    __tablename__ = "XHS_COLLECTION_NAME"
    
    note_id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(255), nullable=False) 
    publish_time = Column(Date)
    user_nickname = Column(String(100)) 