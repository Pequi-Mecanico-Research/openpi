#!/usr/bin/env python3
"""
Ponte ROS 2 para a imagem de uma câmera do manipulador.

Publica os frames que o main.py já captura (os mesmos da observação enviada à policy) como
sensor_msgs/Image. Assim, quem precisa da imagem (ex.: a interface web do manipulator_orchestration,
que repassa o tópico ao YOLO-World) não abre a RealSense de novo: uma câmera só pode ser aberta por um processo.

Como a observação só é lida a cada chunk de ações (~1,7 s), o main.py também chama `publish` a cada passo
do loop com o último frame da câmera; o `max_hz` limita a taxa real de publicação.

Teste manual (envia um padrão de teste, sem câmera):
    python ros2_camera_bridge.py --topic /table_arm/env_camera/image_raw
"""

import argparse
import logging
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)


class CameraImageBridge:
    """Nó ROS 2 que publica frames HxWx3 uint8 de uma câmera como sensor_msgs/Image."""

    def __init__(
        self,
        topic: str = "/table_arm/env_camera/image_raw",
        frame_id: str = "cam_low",
        encoding: str = "rgb8",
        max_hz: float = 10.0,
        node_name: str = "openpi_camera_bridge",
    ):
        # Importados aqui para que os modos sem ROS 2 instalado continuem rodando.
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.signals import SignalHandlerOptions
        from sensor_msgs.msg import Image

        self._rclpy = rclpy
        self._image_cls = Image
        self._encoding = encoding
        self._frame_id = frame_id
        self._min_period = 1.0 / max_hz if max_hz > 0 else 0.0
        self._lock = threading.Lock()
        self._last_publish = 0.0
        self._closed = False

        # O rclpy.init é por processo: se a BaseVelocityBridge já iniciou, reaproveita o contexto
        # (e quem o criou é quem o encerra). Sem o handler de Ctrl+C do ROS: o main.py trata o KeyboardInterrupt.
        self._owns_context = not rclpy.ok()
        if self._owns_context:
            rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        self._node = rclpy.create_node(node_name)
        self._publisher = self._node.create_publisher(Image, topic, 1)  # reliable, depth 1: igual à assinatura do painel

        # Sem timers/callbacks, mas o executor mantém o nó descobrível (matching de publisher/subscriber).
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()
        logger.info(f"ROS 2 camera bridge publishing {frame_id} on {topic} ({encoding}, max {max_hz:g} Hz)")

    def ready(self) -> bool:
        """True se já passou o intervalo do `max_hz` (evita ler o frame à toa)."""
        return not self._closed and time.monotonic() - self._last_publish >= self._min_period

    def publish(self, frame: np.ndarray, force: bool = False) -> bool:
        """Publica um frame HxWx3 uint8 (na ordem de canais de `encoding`). Retorna False se descartado pelo `max_hz`."""
        if self._closed:
            return False
        now = time.monotonic()
        with self._lock:
            if not force and now - self._last_publish < self._min_period:
                return False
            self._last_publish = now
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"expected an HxWx3 image, got shape {frame.shape}")
        msg = self._image_cls()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        msg.height, msg.width = frame.shape[:2]
        msg.encoding = self._encoding
        msg.is_bigendian = 0
        msg.step = frame.shape[1] * 3
        msg.data = frame.tobytes()
        self._publisher.publish(msg)
        return True

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown()
        self._spin_thread.join(timeout=2.0)  # Espera a thread sair antes de destruir o nó.
        self._node.destroy_node()
        if self._owns_context:
            self._rclpy.shutdown()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Publish a moving test pattern for a few seconds")
    parser.add_argument("--topic", default="/table_arm/env_camera/image_raw", help="sensor_msgs/Image topic")
    parser.add_argument("--duration", type=float, default=10.0, help="Seconds to publish before stopping")
    args = parser.parse_args()

    bridge = CameraImageBridge(topic=args.topic, frame_id="test_pattern")
    try:
        end = time.monotonic() + args.duration
        image = np.zeros((480, 640, 3), np.uint8)
        while time.monotonic() < end:
            image[:] = (int((time.monotonic() * 60) % 255), 80, 160)
            bridge.publish(image)
            time.sleep(1 / 30)
    finally:
        bridge.close()
