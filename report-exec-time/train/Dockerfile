# FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:1.14.0-cpu-py36-ubuntu16.04

# AttributeError: module 'tensorflow' has no attribute 'get_default_graph'

#FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.0.0-cpu-py36-ubuntu18.04

FROM tensorflow/tensorflow

RUN apt-get update && \
    apt-get install -y curl 

RUN pip install numpy pandas --upgrade

# RUN pip install gevent gunicorn flask

ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY code /opt/program
WORKDIR /opt/program
