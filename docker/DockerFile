# docker build --no-cache=true -t smith_hyperparams1_demo -f DockerFile .

FROM tensorflow/tensorflow:1.8.0-py3

RUN mkdir -p /tmp/setup && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        libxpm4 \
        libtbb2 \
        tar \
        wget \
    && rm -rf /var/lib/apt/lists/*

RUN cd /usr/local && \
    git clone https://github.com/GilesStrong/Smith_HyperParams1_Demo /Smith_HyperParams1_Demo && \
    mkdir /Smith_HyperParams1_Demo/Data && \
    wget -O /Smith_HyperParams1_Demo/Data/atlas-higgs-challenge-2014-v2.csv.gz http://opendata.cern.ch/record/328/files/atlas-higgs-challenge-2014-v2.csv.gz && \
	gunzip /Smith_HyperParams1_Demo/Data/atlas-higgs-challenge-2014-v2.csv.gz

WORKDIR /Smith_HyperParams1_Demo

ADD . /Smith_HyperParams1_Demo

RUN pip install --trusted-host pypi.python.org matplotlib==2.1.2 pandas==0.22.0 six numpy==1.14.0 seaborn==0.8.1 jupyter scikit-learn==0.19.1 tensorflow==1.8.0 keras==2.1.6

ENV KERAS_BACKEND tensorflow

EXPOSE 8888

CMD ["/run_jupyter.sh", "--allow-root"]