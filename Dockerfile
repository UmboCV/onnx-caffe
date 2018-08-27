########################################
FROM ubuntu:18.04 as caffe
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
        wget \
        libatlas-base-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libprotobuf-dev \
        protobuf-compiler  \
 &&    rm -rf /var/lib/apt/lists/*

########################################
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ENV PATH /opt/conda/bin:$PATH


# Pre-requisite for building pytorch from source,
# use default boost with conda (tried v1.60, cause boost-python signature error)
RUN conda install -y boost \
 && conda clean -ya

# ENV CAFFE_ROOT=/opt/caffe
# WORKDIR $CAFFE_ROOT
# 
# # FIXME: use ARG instead of ENV once DockerHub supports this
# # https://github.com/docker/hub-feedback/issues/460
# ENV CLONE_TAG=master
# ENV nproc=4
# 
# RUN git clone -b ${CLONE_TAG} --depth 1 https://github.com/BVLC/caffe.git . && \
#     cd python && for req in $(cat requirements.txt) pydot; do pip install $req; done && cd .. && \
#     pip install --upgrade pip pytest python-dateutil 
# 
# RUN mkdir build && cd build && \
#     cmake -DCPU_ONLY=1 -DUSE_OPENCV=0 -DBUILD_docs=0 -DUSE_LEVELDB=0 -DUSE_LMDB=0 -DUSE_HDF5=1  -DBUILD_SHARED_LIBS=0 -DBOOST_ROOT=/opt/conda/ ..
#     make -j"$(nproc)"
# 
# ENV PYCAFFE_ROOT $CAFFE_ROOT/python
# ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
# ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
# RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

WORKDIR /workspace
