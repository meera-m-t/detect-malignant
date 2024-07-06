FROM mambaorg/micromamba:latest as builder

COPY docker-environment.yml .

RUN micromamba create --file docker-environment.yml


FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

EXPOSE 8888



SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update \
    && apt-get install curl make g++ git -y \
    && useradd -ms /bin/bash cloudcorn

COPY --from=builder /opt/conda /opt/conda

COPY --from=builder /bin/micromamba /usr/bin/micromamba

COPY --from=builder /usr/local/bin/_activate_current_env.sh /usr/local/bin/_activate_current_env.sh

ENV MAMBA_EXE=/usr/bin/micromamba
ENV CONDA_PREFIX=/opt/conda/envs/my-docker-environment
ENV MAMBA_ROOT_PREFIX=/opt/conda/

ADD . /home/cloudcorn/app

RUN chown -R cloudcorn:cloudcorn /home/cloudcorn/app


WORKDIR /home/cloudcorn/app


USER cloudcorn

RUN echo source /usr/local/bin/_activate_current_env.sh >> ~/.bashrc \
    && eval "$(micromamba shell hook shell=bash)" \
    && micromamba activate my-docker-environment \
    && chmod +x docker-entrypoint.sh

EXPOSE 8888

ENTRYPOINT ["./docker-entrypoint.sh"]