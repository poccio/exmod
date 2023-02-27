FROM nvidia/cuda:11.8.0-base-ubuntu22.04

WORKDIR /root

# install utilities

RUN \
    DEBIAN_FRONTEND="noninteractive" apt-get update && \
    DEBIAN_FRONTEND="noninteractive" apt-get install -y wget sudo

EXPOSE 8000

# install conda
RUN \
    wget -O miniconda.sh "https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh" && \
    bash miniconda.sh -b -p /root/miniconda3 && \
    rm -f miniconda.sh
ENV PATH=/root/miniconda3/bin:${PATH}
RUN conda update -y conda && conda init

# setup env
WORKDIR /exmod
COPY setup.sh requirements.txt ./
RUN \
    bash -c "source ~/miniconda3/etc/profile.d/conda.sh && printf 'exmod\n3.9\n1.11.0\n11.3\n1.7.3\n' | bash setup.sh"
COPY . .

# standard cmd
CMD ["bash", "-c", "source ~/miniconda3/etc/profile.d/conda.sh && conda activate exmod && PYTHONPATH=$(pwd) classy serve experiments/bart-semcor-wne-fews/2023-02-25/14-30-47/checkpoints/best.ckpt -p 8000"]
