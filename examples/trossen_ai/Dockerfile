# syntax=docker/dockerfile:1
# Cliente Trossen AI <-> OpenPI para o braço WidowX AI + base móvel.
# ROS 2 Jazzy (Ubuntu 24.04) com Python 3.12: o rclpy é importado direto pelo main.py.
#
# Inferência (braço + base), a partir da raiz do repositório:
#   ./client_inference_base.sh --task_prompt "move the cube from the table"
#
# Shell no container para rodar outros comandos (a partir desta pasta, examples/trossen_ai):
#   docker compose run --rm --build trossen_openpi_base

FROM ros:jazzy-ros-base
COPY --from=ghcr.io/astral-sh/uv:0.11.19 /uv /uvx /bin/

# git: dependência lerobot_trossen vem do GitHub. libgl1/libglib2.0-0: opencv. libusb: câmeras RealSense.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential libgl1 libglib2.0-0 libusb-1.0-0 ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*

# O PyPI só tem wheels x86_64 do pyrealsense2 (o pyproject.toml o restringe a x86_64): no aarch64
# compila o librealsense com os bindings Python, instalados no Python do sistema (visível no venv).
# Backend RSUSB: acessa as câmeras via libusb, sem patches no kernel do host.
ARG TARGETARCH
ARG LIBREALSENSE_VERSION=2.56.5
RUN if [ "$TARGETARCH" = "arm64" ]; then \
        apt-get update && apt-get install -y --no-install-recommends \
            cmake pkg-config python3-dev libssl-dev libusb-1.0-0-dev libudev-dev \
        && git clone --depth 1 --branch v${LIBREALSENSE_VERSION} \
            https://github.com/IntelRealSense/librealsense.git /tmp/librealsense \
        && cmake -S /tmp/librealsense -B /tmp/librealsense/build \
            -DCMAKE_BUILD_TYPE=Release \
            -DBUILD_PYTHON_BINDINGS=ON \
            -DPYTHON_EXECUTABLE=/usr/bin/python3.12 \
            -DPYTHON_INSTALL_DIR=/usr/lib/python3/dist-packages/pyrealsense2 \
            -DFORCE_RSUSB_BACKEND=ON \
            -DBUILD_EXAMPLES=OFF -DBUILD_GRAPHICAL_EXAMPLES=OFF -DBUILD_TOOLS=OFF \
        && cmake --build /tmp/librealsense/build -j"$(nproc)" \
        && cmake --install /tmp/librealsense/build \
        && ldconfig \
        && rm -rf /tmp/librealsense /var/lib/apt/lists/* \
        && python3.12 -c "import pyrealsense2"; \
    fi

ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# Mesma estrutura do repositório, pois o uv.lock referencia ../../packages/openpi-client.
WORKDIR /app/examples/trossen_ai
COPY --from=openpi_client . /app/packages/openpi-client

# venv com o Python 3.12 do sistema (o mesmo do ROS 2 Jazzy) e acesso aos pacotes do sistema,
# para enxergar rclpy/geometry_msgs de /opt/ros/jazzy (adicionados ao PYTHONPATH pelo setup.bash).
RUN uv venv --python /usr/bin/python3.12 --system-site-packages $UV_PROJECT_ENVIRONMENT
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY . .

# `docker exec` não passa pelo /ros_entrypoint.sh: carrega o ROS 2 (ros2, rclpy) em todo shell bash,
# interativo (.bashrc, ex.: `docker exec -it <container> bash`) ou não (BASH_ENV, ex.: `bash -c "ros2 ..."`).
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" > /etc/ros_env.sh \
    && echo "source /etc/ros_env.sh" >> /root/.bashrc
ENV BASH_ENV=/etc/ros_env.sh
# Mesmo RMW da base (Cyclone DDS).
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
# `uv run` não re-sincroniza o venv já instalado na imagem (use `uv run` ou `python` direto).
ENV UV_NO_SYNC=1

# /ros_entrypoint.sh só carrega o ambiente do ROS 2 e executa o comando recebido.
# Sem comando abre um shell; o main.py é iniciado pelo client_inference_base.sh na raiz do repositório.
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
