# Using the official tensorflow serving image from docker hub as base image
FROM tensorflow/serving

# Installing NGINX, used to rever proxy the predictions from SageMaker to TF Serving
RUN apt-get update && apt-get install -y --no-install-recommends nginx git

# Copy our model folder to the container
COPY saved_model_half_plus_three /saved_model_half_plus_three

# Copy NGINX configuration to the container
COPY nginx.conf /etc/nginx/nginx.conf

# starts NGINX and TF serving pointing to our model
ENTRYPOINT service nginx start | tensorflow_model_server --rest_api_port=8501 \
 --model_name=half_plus_three \
 --model_base_path=/saved_model_half_plus_three

