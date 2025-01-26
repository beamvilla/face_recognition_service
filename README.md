# face_recognition_service

I implemented code from https://medium.com/datawiz-th/%E0%B8%97%E0%B8%B3-face-recognition-%E0%B8%94%E0%B9%89%E0%B8%A7%E0%B8%A2%E0%B8%A7%E0%B8%B4%E0%B8%98%E0%B8%B5-n-shot-learning-%E0%B8%9A%E0%B8%99-pytorch-956e90a373a9

# Build image
```
docker build -f docker/Dockerfile -t face_recognition_service:latest .
```


# Test
```
curl -X 'POST' \
  'http://localhost:8000/face_recognize' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'image=@dataset/face/test/suzy/images.jpg;type=image/jpeg'
```