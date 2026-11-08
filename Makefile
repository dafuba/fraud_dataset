PY ?= python

.PHONY: setup train test notebook report api dashboard drift clean

setup:            ## create the conda env and install the package
	conda env create -f environment.yml || conda env update -f environment.yml
	$(PY) -m ipykernel install --user --name fraud-detection --display-name "Python (fraud-detection)"

train:            ## train both models and write models/ + model card
	$(PY) scripts/train.py

test:             ## unit + API tests
	$(PY) -m pytest -q

notebook:         ## execute the notebook in place
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=fraud-detection proj.ipynb

report: notebook  ## static HTML version of the notebook
	$(PY) -m jupyter nbconvert --to html --output-dir reports proj.ipynb

api:              ## run the scoring API on :8000
	$(PY) -m uvicorn app.api:app --reload

dashboard:        ## run the analyst console
	$(PY) -m streamlit run app/dashboard.py

drift:            ## PSI check of a batch against training data (make drift FILE=batch.csv)
	$(PY) scripts/check_drift.py $(FILE)

clean:
	rm -rf models/*.joblib models/model_card.json reports/*.html
