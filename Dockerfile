FROM debian:latest

RUN apt-get -y update && apt-get install -y git python3-pip python3-dev python3-tk vim procps curl libsm6 libxext6


ADD . /usr/share/face_classification/

WORKDIR /usr/share/face_classification/src
RUN pip3 install -r ../requirements.txt


ENV PYTHONPATH=$PYTHONPATH:src

ENTRYPOINT ["python3"]
CMD ["video_emotion_color_demo.py"]

# # docker run -ti --device=/dev/video0:/dev/video0 -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix/:/tmp/.X11-unix CLIENT_ID

