import io
import numpy as np
import uvicorn
from fastapi import FastAPI, File
from PIL import Image
from src.recognition import NumpbersRecognition


recognition = NumpbersRecognition()

app = FastAPI()
@app.post('/recognition')
def post_recognition(file: bytes = File(...)):
    results = {}
    results['status'] = 1
    image = Image.open(io.BytesIO(file)).convert("RGB")
    image = np.array(image)
    image = image[:,:,::-1].copy()
    results['recognition'] = recognition(image)
    return results

if __name__ == '__main__':
   uvicorn.run("main:app", host="0.0.0.0", port=8080)

