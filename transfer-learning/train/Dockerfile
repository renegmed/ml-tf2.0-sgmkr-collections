# FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:1.14.0-cpu-py36-ubuntu16.04

# AttributeError: module 'tensorflow' has no attribute 'get_default_graph'
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.0.0-cpu-py36-ubuntu18.04

RUN apt-get update && \
    apt-get install -y curl

# RUN pip install gevent gunicorn flask

RUN pip install --upgrade pip

ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY code /opt/program
WORKDIR /opt/program



#RUN apt-get update && apt-get install -y --no-install-recommends curl boto3 numpy sagemaker

#RUN wget https://bootstrap.pypa.io/get-pip.py && python3.6 get-pip.py && \
#    pip3 install sagemaker==1.50.0 numpy pandas boto3 botocore && \
#        (cd /usr/local/lib/python3.6/dist-packages/scipy/.libs; rm *; ln ../../numpy/.libs/* .) && \
#        rm -rf /root/.cache
        
#RUN apt-get update && apt-get install -y --no-install-recommends nginx curl

#RUN pip install gevent gunicorn

#ENV PATH="/opt/ml/code:${PATH}"



