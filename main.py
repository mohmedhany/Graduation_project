import io
from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from starlette.responses import StreamingResponse

model = tf.keras.models.load_model('model_over600.h5', compile=False)

app = FastAPI()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img_io = io.BytesIO(img_bytes)
    img_pil = Image.open(img_io)
    img_np = np.array(img_pil)
    img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    resized_img = cv2.resize(img_cv2, (256, 256))
    rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    normalized_img = rgb_img.astype('float32') / 255.0

    if len(normalized_img.shape) == 3:  # Check if it has 3 channels (RGB)
        normalized_img = np.expand_dims(normalized_img, axis=0)

    predictions = model.predict(normalized_img)
    output_image = Image.fromarray((predictions[0] * 255).astype(np.uint8))
    output_bytes = io.BytesIO()
    output_image.save(output_bytes, format='JPEG')
    output_bytes.seek(0)
    return StreamingResponse(content=output_bytes, media_type="image/jpeg")


if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="127.0.0.1")
