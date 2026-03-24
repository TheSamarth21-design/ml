import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    model = joblib.load("model.pkl")
    print("✅ NASA Brain Loaded Successfully!")
except Exception as e:
    print(f"❌ Error: model.pkl not found! {e}")

BASELINE = [518.67, 642.35, 1589.70, 1400.60, 14.62, 21.61, 554.36, 2388.06, 9046.19, 1.30, 47.47, 521.66, 2388.02, 8138.62, 8.41, 0.03, 392, 2388, 100, 39.06, 23.41]

@app.get("/")
def home():
    return {"status": "AI API is Online"}

@app.post("/predict-file")
async def predict_from_file(file: UploadFile = File(...)):
    contents = await file.read()
    filename = file.filename.lower()
    
    # STRICTLY CSV ONLY
    if not filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Strictly CSV files are supported. Please upload a .csv")

    try:
        # Read the CSV (with protections against weird Mac characters)
        df_raw = pd.read_csv(io.BytesIO(contents), header=None, encoding='latin1', on_bad_lines='skip')

        # Extract Row
        data_row = df_raw.iloc[0].values.tolist()
        if isinstance(data_row[0], str):
            data_row = df_raw.iloc[1].values.tolist()

        # Extract 21 sensors
        sensor_data = []
        for i in range(21):
            try:
                val = float(data_row[i + 5])
                sensor_data.append(val if not pd.isna(val) else BASELINE[i])
            except:
                sensor_data.append(BASELINE[i])

        # Predict
        df_model = pd.DataFrame([sensor_data], columns=[f'sensor_{i}' for i in range(1, 22)])
        prediction = int(model.predict(df_model)[0])
        
        return {"rul": prediction}

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        raise HTTPException(status_code=400, detail=f"CSV Processing Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    