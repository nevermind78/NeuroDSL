# CPU-only, CUDA-free demonstration image for one notebook:
# pumas_dyad_locality_demo.ipynb. Builds a minimal NeuroDSL environment
# (JSON + Julia stdlib only -- see NeuroDSL_pkg/Project.toml for why CUDA,
# Flux, Zygote etc. are safely omitted) so the image is small and does not
# require a GPU. Follows the BinderHub Dockerfile spec (jovyan user, home
# directory as working directory) so this also launches directly on
# mybinder.org from this repo.
FROM julia:1.10

ARG NB_USER=jovyan
ARG NB_UID=1000
ENV USER=${NB_USER}
ENV HOME=/home/${NB_USER}
ENV JULIA_DEPOT_PATH=/home/${NB_USER}/.julia

RUN useradd -m -s /bin/bash -N -u ${NB_UID} ${NB_USER}

# Jupyter server itself (IJulia provides the Julia kernel, not the notebook
# server/UI) -- kept to the one small package actually needed.
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip \
    && pip3 install --no-cache-dir --break-system-packages notebook \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=${NB_UID} NeuroDSL_pkg ${HOME}/NeuroDSL_pkg
COPY --chown=${NB_UID} Project.toml ${HOME}/Project.toml
COPY --chown=${NB_UID} pumas_dyad_locality_demo.ipynb ${HOME}/pumas_dyad_locality_demo.ipynb

USER ${NB_USER}
WORKDIR ${HOME}

RUN julia --project=. -e ' \
    using Pkg; \
    Pkg.develop(path="NeuroDSL_pkg"); \
    Pkg.instantiate(); \
    Pkg.build("IJulia"); \
    using NeuroDSL; \
    println("NeuroDSL loaded OK. CUDA_AVAILABLE = ", NeuroDSL.Backend.CUDA_AVAILABLE)'

EXPOSE 8888
CMD ["jupyter", "notebook", "--ip", "0.0.0.0", "--port", "8888", "--no-browser", \
     "--NotebookApp.token=''", "--NotebookApp.password=''"]
