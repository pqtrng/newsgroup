import contextlib
import os
import joblib
from fastapi import Depends, FastAPI
from pydantic import BaseModel
from sklearn.pipeline import Pipeline


class PredictionInput(BaseModel):
    text: str


class PredictionOutput(BaseModel):
    category: str


class NewsgroupsModel:
    model: Pipeline | None = None
    targets: list[str] | None = None

    def load_model(self):
        """Load the model from the model file."""
        model_file = os.path.join(os.path.dirname(__file__), "newsgroups_model.joblib")
        loaded_model: tuple[Pipeline, list[str]] = joblib.load(model_file)
        model, targets = loaded_model
        self.model = model
        self.targets = targets

    def predict(self, input: PredictionInput) -> PredictionOutput:
        """Run a prediction on the input text."""
        if not self.model or not self.targets:
            raise RuntimeError("Model is not loaded")

        prediction = self.model.predict([input.text])
        category = self.targets[prediction[0]]
        return PredictionOutput(category=category)


newsgroup_model = NewsgroupsModel()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    newsgroup_model.load_model()
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/prediction")
async def prediction(
    output: PredictionOutput = Depends(newsgroup_model.predict),
) -> PredictionOutput:
    return output
