FROM intelpython/intelpython3_core
RUN apt update
RUN apt install vim -y
ENV LANG C.UTF-8
RUN mkdir -p /repos/github/
RUN git clone https://github.com/mmngreco/aguathon_2019.git /root/repos/aguathon
RUN conda env create -f /root/repos/aguathon/environment.yml
RUN echo "conda activate river" >> /root/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH
WORKDIR /root/repos/aguathon
