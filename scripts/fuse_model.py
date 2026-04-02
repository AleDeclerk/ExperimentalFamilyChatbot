"""
Fusiona el adapter LoRA con el modelo base para generar un modelo completo.

Uso:
    python scripts/fuse_model.py \
        --base-model tiiuae/Falcon-H1R-7B \
        --adapter-path models/adapters/v1 \
        --output-path models/fused/v1
"""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Fusionar adapter LoRA con modelo base")
    parser.add_argument("--base-model", type=str, default="tiiuae/Falcon-H1R-7B")
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()

    print(f"Cargando modelo base {args.base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    print(f"Cargando adapter de {args.adapter_path}...")
    model = PeftModel.from_pretrained(model, str(args.adapter_path))

    print("Fusionando adapter con modelo base...")
    model = model.merge_and_unload()

    print(f"Guardando modelo fusionado en {args.output_path}...")
    args.output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output_path))
    tokenizer.save_pretrained(str(args.output_path))
    print("Fusión completada.")


if __name__ == "__main__":
    main()
