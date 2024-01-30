FROM nvidia/cuda:12.2.0-base-ubuntu22.04
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# patch security packages
RUN printf '#!/bin/sh\nexit 0' > /usr/sbin/policy-rc.d

RUN apt update; apt upgrade; apt install -y software-properties-common; add-apt-repository -y ppa:deadsnakes/ppa

ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update; apt-get -s dist-upgrade | grep "^Inst" | grep -iE 'securi|ssl|login|passwd' | awk -F " " {'print $2'} | xargs apt-get install \
    && apt-get install -y python3.8 python3.8-distutils python3.8-dev curl build-essential git libsystemd0 systemd udev systemd-container systemd-coredump systemd-tests systemd-journal-remote libudev-dev libsystemd-dev libnss-resolve libnss-mymachines* libnss-myhostname libc-bin uidmap \
    # install dependencies via apt
    && rm -rf /var/lib/{apt,dpkg,cache,log}

RUN apt-get install ffmpeg libsm6 libxext6  -y
# add host
RUN mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts

# install python dependencies
RUN curl "https://bootstrap.pypa.io/get-pip.py" -o "get-pip.py" && \
    python3.8 get-pip.py && \
    python3.8 -m pip install -r requirements.txt --ignore-installed && \
    python3.8 setup.py build_ext --inplace && \
    python3.8 -m nltk.downloader stopwords && \
    python3.8 -m nltk.downloader punkt && \
    python3.8 -m pip install torch torchvision torchaudio --ignore-installed

EXPOSE 5000
CMD gunicorn -b 0.0.0.0:5000 --name nlm-model-server --workers 1 --timeout 3600 modelserver_restserver:app
