FROM tensorflow/serving:latest
 
COPY ./serving_model_dir /models
ENV MODEL_NAME=predict-student-performance-model