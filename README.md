DeepFace : Face Recognition and Facial attribute analysis Framework

>>> reference : https://github.com/serengil/deepface

===============

### Installation ###
***
1. conda create -n “name” python=3.9.12
2. source activate “name”
3. git clone 
4. cd ailab-face-analysis
5. pip install .
6. conda install tensorflow-gpu


### Docker Setting ###
***
1. git clone 
2. cd ailab-face-analysis
3. docker build -t "image_name":"tag" .
4. docker run -p 8800:8800 -d --name "container_name" "image_name":"tag"

### client.py ###
***

    import base64
    import json                    
    import requests
    import time
    url = "http://0.0.0.0:8800/items/"
    
    image_file = 'image.jpg'
    def get_face_info(image_file: str):
        with open(image_file, "rb") as f:
            im_bytes = f.read()        
    
        im_b64 = base64.b64encode(im_bytes).decode("utf8")
        payload = json.dumps({"image": im_b64})
    
        s = time.time()
        response = requests.post(url, data=payload)
        print(f"response time : {int((time.time() - s) * 1000)}ms")
        try:
            data = response.json()     
            print(data)
        except requests.exceptions.RequestException:
            print(response.text)
            
    if __name__ == "__main__":
        get_face_info(image_file)
