FROM python:3.10 AS libtorch

RUN pip install torch

RUN LIBTORCH=$(python -c 'import torch; from pathlib import Path; print(Path(torch.__file__).parent)') && \
    mkdir /libtorch && \
    cp -r $LIBTORCH/lib /libtorch/ && \
    cp -r $LIBTORCH/include /libtorch/

FROM rust:latest

COPY --from=libtorch /libtorch /libtorch

RUN apt update && apt install -y cmake

ENV LIBTORCH /libtorch
ENV LD_LIBRARY_PATH /libtorch/lib
ENV LIBTORCH_CXX11_ABI 0

WORKDIR /syntaxdot
ADD . .
RUN cargo build --release

WORKDIR /patchelf
RUN wget "https://github.com/NixOS/patchelf/releases/download/0.12/patchelf-0.12.tar.bz2" && \
    echo "699a31cf52211cf5ad6e35a8801eb637bc7f3c43117140426400d67b7babd792  patchelf-0.12.tar.bz2" | sha256sum -c - && \
    tar jxf patchelf-0.12.tar.bz2 && \
    ( cd patchelf-0.12.20200827.8d3a16e && ./configure && make -j4 )

WORKDIR /dist
RUN cp /syntaxdot/target/release/syntaxdot .
RUN cp /libtorch/lib/*.so* .
RUN /patchelf/patchelf-0.12.20200827.8d3a16e/src/patchelf --set-rpath '$ORIGIN' *.so* syntaxdot

ENTRYPOINT [ "/dist/syntaxdot" ]