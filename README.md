# face_recognition_service

I implemented code from https://medium.com/datawiz-th/%E0%B8%97%E0%B8%B3-face-recognition-%E0%B8%94%E0%B9%89%E0%B8%A7%E0%B8%A2%E0%B8%A7%E0%B8%B4%E0%B8%98%E0%B8%B5-n-shot-learning-%E0%B8%9A%E0%B8%99-pytorch-956e90a373a9

# For training model
## Build model
```
docker build -f docker/Dockerfile -t face_recognition_service:latest .
```
## Start enviroment
```
docker run -it --net=host --rm -v $(pwd):/face_recognition_service face_recognition_service:latest /bin/bash
```
## Train
### 1. Edit train config in /config/train.yaml
### 2. Run
```
python3 ./src/train.py
```
### 3. Mock face features db
```
python3 ./src/store_face_features.py
```

# For inference
## Start docker
```
docker-compose -f docker-compose.yml up

# If want to re-build image
docker-compose -f docker-compose.yml up --build
```
## Test result
```
curl -X 'POST' \
  'http://localhost:8000/face_recognize' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'image=@dataset/face/test/suzy/images.jpg;type=image/jpeg'

# [NOTE] 'dataset/face/test/suzy/images.jpg' is the path of image
```