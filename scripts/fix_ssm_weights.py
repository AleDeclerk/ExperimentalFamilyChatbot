"""
Repara tensores SSM con valores inf/nan en el modelo fusionado antes de convertir a GGUF.
Clamp los valores a un rango seguro.
"""

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main():
    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("models/fused/v1")

    shard_files = sorted(model_dir.glob("*.safetensors"))
    if not shard_files:
        print(f"No se encontraron archivos safetensors en {model_dir}")
        sys.exit(1)

    for shard_path in shard_files:
        print(f"Procesando {shard_path.name}...")
        tensors = load_file(str(shard_path))
        modified = False

        for name, tensor in tensors.items():
            has_inf = torch.isinf(tensor).any().item()
            has_nan = torch.isnan(tensor).any().item()
            if has_inf or has_nan:
                print(f"  REPARANDO {name}: inf={has_inf}, nan={has_nan}, "
                      f"shape={list(tensor.shape)}, dtype={tensor.dtype}")
                # Reemplazar nan con 0 e inf con valores grandes pero finitos
                tensor = torch.nan_to_num(tensor, nan=0.0, posinf=65504.0, neginf=-65504.0)
                tensors[name] = tensor
                modified = True

        if modified:
            print(f"  Guardando {shard_path.name} reparado...")
            save_file(tensors, str(shard_path))
        else:
            print(f"  OK, sin valores problemáticos.")

    print("Done.")


if __name__ == "__main__":
    main()
