import os
import joblib

model_file = os.path.join(os.path.dirname(__file__), "newsgroups_model.joblib")
model, targets = joblib.load(model_file)

p = model.predict(["God is love"])
print(targets[p[0]])
