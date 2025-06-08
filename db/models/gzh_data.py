from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Date, Float, BigInteger, Text
from sqlalchemy.ext.declarative import declarative_base
import os

load_dotenv()

Base = declarative_base()

class GZHData(Base):
    __tablename__ = "GZH_COLLECTION_NAME"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(255), nullable=False)
    description = Column(Text)
    publish_time = Column(Date)
    user_nickname = Column(String(100))
    content_url = Column(String(100)) 