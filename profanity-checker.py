import joblib
import numpy as np

vectorizer = joblib.load('./data/vectorizer.joblib')
model = joblib.load('./data/model.joblib')


def _get_profane_prob(prob):
    return prob[1]


def predict(texts):
    return model.predict(vectorizer.transform(texts))


def predict_prob(texts):
    return np.apply_along_axis(_get_profane_prob, 1, model.predict_proba(vectorizer.transform(texts)))

# If you cock a part of your body in a particular direction, you lift it or point it in that direction. 0
# You refer to a male bird, especially a male game bird, as a cock when you want to distinguish it from a female bird 0
# He paused and cocked his head as if listening. 0
# The Brigadier thought about this for a moment, head cocked to one side. 0
# If someone cocks their ear, they try very hard to hear something from a particular direction. 0
# His hands were too weak to cock his revolver 1
# A cock is an adult male chicken.
print(predict(["I am here"]))
