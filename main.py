import base64
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from deepface import DeepFace
from PIL import Image
import io
import numpy as np
from pydantic import BaseModel

from cam import get_stream_video

key_list = ('region',
            'dominant_emotion',
            'gender',
            'age',
            'dominanat_race')

class Item(BaseModel):
    image: str

def video_streaming():
    return get_stream_video()


app = FastAPI()
@app.get("/video")
def main():
    # StringResponse함수를 return하고,
    # 인자로 OpenCV에서 가져온 "바이트"이미지와 type을 명시
    return StreamingResponse(video_streaming(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post('/items/')
async def create_item(image: Item):
    # result = {'image' : image.image}
    faces = {}
    try:
             
        # get the base64 encoded string
        im_b64 = image.image

        # convert it into bytes  
        img_bytes = base64.b64decode(im_b64.encode('utf-8'))

        # convert bytes data to PIL Image object
        img = Image.open(io.BytesIO(img_bytes))

        # PIL image object to numpy array
        img_arr = np.asarray(img)
        demography = DeepFace.analyze(img_path = img_arr,
            actions=['emotion', 'age', 'gender'],
            detector_backend = 'retinaface',
            enable_multiple=True,
            prog_bar=False
            )
        for face, val in demography.items():
            face_info = {}
            for k, v in val.items():
                if k in key_list:
                    face_info[k] = v
    
            faces[face] = face_info
    except Exception as e:
        faces['error'] = e
    return faces

