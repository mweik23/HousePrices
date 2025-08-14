
'''
development usage: uvicorn app.main:app --reload

production usage (1 worker):  uvicorn app.main:app --host 0.0.0.0 --port 8000
'''

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import numpy as np
import pandas as pd
import joblib
import io
import time
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.append(str(PROJECT_ROOT / "src"))
from HousePrices.utils.io import get_data
model_base_path = PROJECT_ROOT / "registry"

#---------- define model info-----------
model_name = 'HousePrices_regressor'
model_version = 'v1'
#---------------------------------------

out_var = 'SalePrice'  # Default output variable name

model_path = model_base_path / model_name / model_version / 'pipeline.joblib'

# load model
model = joblib.load(model_path)

# create FastAPI object
app = FastAPI()

# API operations
@app.get("/")
def health_check():
    return {'health_check': 'OK'}

@app.get("/info")
def info():
    return {'name': 'HousePrices', 'description': "Search API for House Prices."}


@app.post("/predict/upload")
async def predict_from_csv(file: UploadFile = File(...)):

    start_time = time.perf_counter()  # high-resolution timer
    if file.content_type not in ("text/csv", "application/vnd.ms-excel"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")
    in_filename = file.filename

    # Read CSV into pandas (without writing to disk)
    content = await file.read()
    try:
        Ids, df, _ = get_data(io.BytesIO(content), load_info=False, print_output=False, print_status=False, split='test')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV parse error: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV has no rows.")
    
    # Run predictions
    try:
        preds = model.predict(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")

    # create dataframe of predictions
    df_preds = pd.DataFrame({
        Ids.name: Ids,
        out_var: preds
    })

    # Convert dataframe to CSV in memory
    buf = io.StringIO()
    df_preds.to_csv(buf, index=False)
    buf.seek(0)

    elapsed = time.perf_counter() - start_time
    print(f"Request processed in {elapsed:.3f} seconds")
    # Stream the CSV back
    out_filename = ''.join(in_filename.split('.')[:-1])+f"_predictions.csv"
    headers = {"Content-Disposition": f'attachment; filename="{out_filename}"'}
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv", headers=headers)

