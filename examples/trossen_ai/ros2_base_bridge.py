#!/usr/bin/env python3
"""
Ponte ROS 2 para a base móvel.

Publica os comandos de velocidade da base (cmd_vel, ang_vel) como geometry_msgs/Twist.
Feita para rodar no container do Dockerfile (ROS 2 Jazzy + Python 3.12), onde o rclpy
é importado direto no mesmo processo do main.py.

Teste manual (dentro do container):
    python ros2_base_bridge.py --topic /cmd_vel --linear 0.1 --angular 0.0 --duration 2
"""

import argparse
import logging
import threading
import time

logger = logging.getLogger(__name__)


class BaseVelocityBridge:
    """Nó ROS 2 que publica as velocidades linear (x) e angular (z) da base."""

    def __init__(self, topic: str = "/cmd_vel", timeout_s: float = 1.0, node_name: str = "openpi_base_bridge"):
        # Importados aqui para que os modos sem base (test/autonomous) rodem sem ROS 2 instalado.
        import rclpy
        from geometry_msgs.msg import Twist
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.signals import SignalHandlerOptions

        self._rclpy = rclpy
        self._twist_cls = Twist
        self._timeout_s = timeout_s
        self._lock = threading.Lock()
        self._last_cmd_time = time.monotonic()
        self._stopped = True
        self._closed = False

        # Sem o handler de Ctrl+C do ROS: o main.py trata o KeyboardInterrupt e para a base no cleanup.
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        self._node = rclpy.create_node(node_name)
        self._publisher = self._node.create_publisher(Twist, topic, 10)
        self._node.create_timer(0.05, self._watchdog)

        # O executor roda o timer do watchdog numa thread separada do loop de controle.
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

        deadline = time.monotonic() + 2.0
        while self._publisher.get_subscription_count() == 0 and time.monotonic() < deadline:
            time.sleep(0.1)
        if self._publisher.get_subscription_count() == 0:
            logger.warning(f"No subscribers on {topic} yet. Is the base driver running?")
        logger.info(f"ROS 2 base bridge publishing on {topic}")

    def _publish(self, linear: float, angular: float):
        msg = self._twist_cls()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self._publisher.publish(msg)

    def _watchdog(self):
        # Segurança: se o loop de controle parar de mandar comandos (travou), para a base uma vez.
        with self._lock:
            if not self._stopped and time.monotonic() - self._last_cmd_time > self._timeout_s:
                logger.warning(f"No base command for {self._timeout_s}s, stopping the base")
                self._publish(0.0, 0.0)
                self._stopped = True

    def send(self, linear: float, angular: float):
        """Publica um comando de velocidade (linear em x, angular em z) na base."""
        with self._lock:
            self._publish(linear, angular)
            self._last_cmd_time = time.monotonic()
            self._stopped = linear == 0.0 and angular == 0.0

    def stop(self):
        """Para a base (velocidade zero). Não lança exceção para poder ser usado em finally/cleanup."""
        if self._closed:
            return
        try:
            self.send(0.0, 0.0)
        except Exception:
            logger.exception("Could not send stop command to the base")

    def close(self):
        if self._closed:
            return
        self.stop()
        self._closed = True
        self._executor.shutdown()
        self._spin_thread.join(timeout=2.0)  # Espera a thread sair antes de destruir o nó.
        self._node.destroy_node()
        self._rclpy.shutdown()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Publish a constant base velocity for a few seconds, then stop")
    parser.add_argument("--topic", default="/cmd_vel", help="geometry_msgs/Twist topic of the base")
    parser.add_argument("--linear", type=float, default=0.1, help="Linear velocity (cmd_vel)")
    parser.add_argument("--angular", type=float, default=0.1, help="Angular velocity (ang_vel)")
    parser.add_argument("--duration", type=float, default=5.0, help="Seconds to publish before stopping")
    args = parser.parse_args()

    bridge = BaseVelocityBridge(topic=args.topic)
    try:
        end = time.monotonic() + args.duration
        while time.monotonic() < end:
            bridge.send(args.linear, args.angular)
            time.sleep(1 / 30)
    finally:
        bridge.close()
