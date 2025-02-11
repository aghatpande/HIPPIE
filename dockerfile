#Pytorch dockerfile
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

USER root

WORKDIR /src

RUN pip install pandas numpy matplotlib scikit-learn pytorch-lightning
RUN pip install wandb
RUN pip install seaborn
#Install nano
RUN apt-get update && apt-get install -y nano

# Copy the dependencies file to the working directory
COPY datasets . 
COPY . .