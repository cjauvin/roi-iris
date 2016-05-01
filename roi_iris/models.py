from sqlalchemy import (
    Column,
    Integer,
    Float,
    Text,
)

from sqlalchemy.ext.declarative import declarative_base

from sqlalchemy.orm import (
    scoped_session,
    sessionmaker,
)

from zope.sqlalchemy import ZopeTransactionExtension

DBSession = scoped_session(
    sessionmaker(extension=ZopeTransactionExtension()))
Base = declarative_base()


class Iris(Base):

    __tablename__ = 'iris'
    iris_id = Column(Integer, primary_key=True, autoincrement=True)
    sepal_length = Column(Float)
    sepal_width = Column(Float)
    petal_length = Column(Float)
    petal_width = Column(Float)
    target_name = Column(Text)

    def __init__(self, sepal_length, sepal_width, petal_length, petal_width, target_name):
        self.sepal_length = sepal_length
        self.sepal_width = sepal_width
        self.petal_length = petal_length
        self.petal_width = petal_width
        self.target_name = target_name
