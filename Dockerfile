FROM ubuntu

LABEL author="Kadir Bulut Ozler"
LABEL description="Container for ling-582 shared task."

RUN apt-get update && apt-get install -y python3-pip

RUN pip install -U torch==1.11.0 \
    torchmetrics \
    transformers==4.17 \
    datasets \
    pandas \
    tqdm \
    numpy \
    spacy \
    nltk \
    clean-text





