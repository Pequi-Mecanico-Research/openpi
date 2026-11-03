import dataclasses

import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_widowx_example() -> dict:
    """Cria um exemplo de entrada aleatório para debug da policy do WidowX AI."""
    return {
        "observation/state": np.random.rand(7),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),      # cam_main
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8), # cam_wrist
        "observation/low_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),   # cam_low
        "prompt": "organize the table",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        import einops

        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class WidowXInputs(transforms.DataTransformFn):
    """
    Converte as entradas do dataset/robô WidowX AI (3 câmeras: cam_main, cam_wrist, cam_low)
    para o formato interno esperado pelo modelo pi0/pi0.5. Usado no treino e na inferência.
    """

    # Determina qual variante do modelo será usada.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # O pi0 suporta 3 slots de imagem: uma visão "base" e duas "wrist".
        base_image = _parse_image(data["observation/image"])        # cam_main -> visão base
        wrist_image = _parse_image(data["observation/wrist_image"])  # cam_wrist -> pulso único
        # Datasets com 2 câmeras (sem cam_low): slot preenchido com zeros e mascarado (mesma convenção do Libero).
        if "observation/low_image" in data:
            low_image = _parse_image(data["observation/low_image"])  # cam_low
            low_mask = np.True_
        else:
            low_image = np.zeros_like(base_image)
            low_mask = np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_

        # Datasets com esforço em coluna separada: concatena ao state (7 pos + 7 esforço = 14),
        # mesmo layout do dataset antigo, que ja trazia os 14 valores em observation.state.
        state = np.asarray(data["observation/state"], dtype=np.float32)
        if "observation/effort" in data:
            state = np.concatenate([state, np.asarray(data["observation/effort"], dtype=np.float32)], axis=-1)
        # Base móvel: odometria (x, y, orientation.z, orientation.w, twist.linear.x, twist.angular.z) no fim do state.
        if "observation/base_odom" in data:
            state = np.concatenate([state, np.asarray(data["observation/base_odom"], dtype=np.float32)], axis=-1)

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": low_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": low_mask,
            },
        }

        # Ações só existem durante o treino.
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class WidowXOutputs(transforms.DataTransformFn):
    """
    Converte a saída do modelo de volta ao formato específico do robô.
    Usado apenas na inferência.
    """

    # 7 = 6 juntas + 1 garra. Datasets com base móvel usam 9 (+ cmd_vel.linear.x, cmd_vel.angular.z).
    action_dim: int = 7

    def __call__(self, data: dict) -> dict:
        print(f"WidowXOutputs: returning only first {self.action_dim} dims of actions")
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}