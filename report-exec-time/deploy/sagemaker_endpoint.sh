#!/usr/bin/env bash

# these commands should be called after create-model command

MODEL_NAME=report-exec-time-v1

ENDPOINT_CONFIG_NAME=report-exec-time-config-v1

ENDPOINT_NAME=report-exec-time-v1

PRODUCTION_VARIANTS="VariantName=Default,ModelName=${MODEL_NAME},"\
"InitialInstanceCount=1,InstanceType=ml.c4.large"

aws sagemaker create-endpoint-config --endpoint-config-name ${ENDPOINT_CONFIG_NAME} \
--production-variants ${PRODUCTION_VARIANTS}

aws sagemaker create-endpoint --endpoint-name ${ENDPOINT_NAME} \
--endpoint-config-name ${ENDPOINT_CONFIG_NAME}



