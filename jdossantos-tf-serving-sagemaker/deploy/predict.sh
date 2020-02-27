#!/usr/bin/env bash 
ENDPOINT_NAME=half-plus-three 

aws sagemaker-runtime invoke-endpoint \
--endpoint-name ${ENDPOINT_NAME} \
--body '{"instances": [1.0,2.0,5.0]}' prediction_response.json 

cat prediction_response.json

