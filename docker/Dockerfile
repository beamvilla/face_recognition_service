FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

WORKDIR /root

RUN apt-get update --yes --quiet \
&& apt-get --yes --quiet upgrade \
&& apt-get install --yes --quiet --no-install-recommends gcc \
&& rm -rf /var/lib/apt/lists/*

ADD /docker/requirements.txt /root/requirements.txt
RUN pip3 install --user --no-cache-dir -r requirements.txt

RUN mkdir -p /face_recognition_service
WORKDIR /face_recognition_service