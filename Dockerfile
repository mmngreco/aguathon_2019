FROM intelpython/intelpython3_core
# RUN apt update
ENV LANG C.UTF-8
RUN mkdir -p /repos/github/
WORKDIR /root/repos
RUN git clone https://github.com/mmngreco/aguathon_2019.git aguathon
RUN conda env create -f aguathon/environment.yml
RUN activate river
