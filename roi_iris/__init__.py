from pyramid.config import Configurator
from prf.view import BaseView

from sqlalchemy import engine_from_config
from .models import DBSession, Base, Iris
import transaction

from sklearn import linear_model, datasets, metrics
import numpy as np


# the order of features is important!
FEATURES = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
TARGET_NAMES = ['setosa', 'versicolor', 'virginica']


# Convert Iris objects to dicts
def row2dict(row):
    return dict((col, getattr(row, col)) for col in row.__table__.columns.keys())


class IrisClassifier:

    """This class maintains a logistic regression model initially trained with
       the sklearn dataset, and eventually with additional user-supplier samples.
       Everything is stored in SQLite to demonstrate the use of a database, but given
       the size of the data, it would make more sense to simply use memory.
    """

    def __init__(self):
        # delete existing content in DB and reload from original dataset
        DBSession.query(Iris).delete()
        d = datasets.load_iris()
        for i, values in enumerate(d.data):
            r = dict(zip(FEATURES, values))
            r['target_name'] = TARGET_NAMES[d.target[i]]
            DBSession.add(Iris(**r))
        transaction.commit()
        self.train()

    def train(self):
        # reload from DB
        rs = [row2dict(r) for r in DBSession.query(Iris).all()]
        self.X = np.asarray([[r[f] for f in FEATURES] for r in rs])
        y = [TARGET_NAMES.index(r['target_name']) for r in rs]
        self.clf = linear_model.LogisticRegression(C=1e5)
        self.clf.fit(self.X, y)
        y_pred = self.clf.predict(self.X)
        # Just verify that the model is not totally bogus (the
        # training error should obviously be quite low)
        assert metrics.accuracy_score(y, y_pred) > 0.95


    def add_sample(self, **kw):
        """Add a new sample, and retrain if desired
        """
        retrain = int(kw.pop('retrain', 0))
        DBSession.add(Iris(**kw))
        transaction.commit()
        # retrain the model with the updated dataset
        if retrain:
            self.train()

    def predict(self, **kw):
        """Predict probs given input vector.
        """
        # use the mean for any unspecified (missing) feature
        vs = [float(kw.get(f, self.X[:, i].mean())) for i, f in enumerate(FEATURES)]
        return dict(zip(TARGET_NAMES, self.clf.predict_proba(np.asarray(vs).reshape(1, -1))[0]))


# Global var.. I know it's not right.. :-(
iris_classifier = None


class IrisView(BaseView):

    def index(self):
        """GET with no params: return current dataset
           GET with feature params (at least one is required): return prediction
        """
        if set(FEATURES) & set(self._params.keys()):
            return iris_classifier.predict(**self._params)
        else:
            return [row2dict(r) for r in DBSession.query(Iris).all()]

    def create(self):
        """POST a new sample, add to database and retrain if specified
        """
        iris_classifier.add_sample(**self._params)


def main(global_config, **settings):
    """ This function returns a Pyramid WSGI application.
    """

    engine = engine_from_config(settings, 'sqlalchemy.')
    DBSession.configure(bind=engine)
    Base.metadata.bind = engine
    Base.metadata.create_all(engine)

    config = Configurator(settings=settings)

    config.include('pyramid_chameleon')
    config.add_static_view('static', 'static', cache_max_age=3600)
    config.add_route('home', '/')

    config.include('prf')
    root = config.get_root_resource()
    root.add('iri', view=IrisView)

    global iris_classifier
    iris_classifier = IrisClassifier()

    config.scan()
    return config.make_wsgi_app()
