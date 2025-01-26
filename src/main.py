import io
from PIL import Image
from fastapi import FastAPI, File, UploadFile

from src.usecase import GetMatchFace
from src.schema import APIResponseBody
from src.config.config import AppConfig


app_config = AppConfig("./config/config.yaml")
get_match_face_usecase = GetMatchFace(app_config)

app = FastAPI()

@app.post(
    "/face_recognize", 
    response_model=APIResponseBody
)
async def recognize(image: UploadFile = File(...)):
    face = Image.open(io.BytesIO(await image.read()))
    match_face = get_match_face_usecase.predict(face)
    return {"success": True, "match_face": match_face}
