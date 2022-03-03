FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime

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

COPY conf/ conf/
COPY dataset/dataloader/ dataset/dataloader/
COPY model/ model/
COPY utils/ utils/
COPY main.py .

RUN pip install -r requirements.txt

RUN rm requirements.txt
