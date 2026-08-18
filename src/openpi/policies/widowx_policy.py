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
        low_image = _parse_image(data["observation/low_image"])      # cam_low 

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": low_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Ações só existem durante o treino.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class WidowXOutputs(transforms.DataTransformFn):
    """
    Converte a saída do modelo de volta ao formato específico do robô.
    Usado apenas na inferência.
    """

    def __call__(self, data: dict) -> dict:
        # retorno de apenas 7 juntas (6 juntas + 1 garra) para o robô WidowX AI.
        print("WidowXOutputs: returning only first 7 dims of actions")
        return {"actions": np.asarray(data["actions"][:, :7])}