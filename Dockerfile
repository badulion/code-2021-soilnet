FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:ubuntugis/ppa && apt-get update && \
    apt-get install -y libgdal-dev g++ --no-install-recommends && \
    apt-get clean -y

## config
ARG USER=dulny
ARG UID=1241

RUN adduser ${USER} --uid ${UID} --home /home/ls6/${USER}/ --disabled-password --gecos "" --no-create-home
RUN mkdir -p /home/ls6/${USER}
RUN chown -R ${USER} /home/ls6/${USER}

USER ${USER}

RUN mkdir -p /home/ls6/${USER}/soilnet-Feb2022/

WORKDIR /home/ls6/${USER}/soilnet-Feb2022/

COPY ./requirements.txt .
RUN pip install -r requirements.txt

RUN rm requirements.txt

COPY conf/ conf/
COPY dataset/dataloader/ dataset/dataloader/
COPY model/ model/
COPY utils/ utils/
COPY main.py .
COPY plot_predictions.py .
COPY zip_vingilot.py .