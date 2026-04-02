"""
Prepare dataset: valida formato JSONL, aplica filtros de calidad y genera split train/test.

Uso:
    python scripts/prepare_dataset.py \
        --input data/raw/conversations.jsonl \
        --output-train data/processed/train.jsonl \
        --output-test data/processed/test.jsonl \
        --test-size 0.1
"""

import argparse
import json
import random
import sys
from pathlib import Path


def validate_message(msg: dict) -> bool:
    """Valida que un mensaje tenga el formato ChatML correcto."""
    if "messages" not in msg:
        return False
    messages = msg["messages"]
    if not isinstance(messages, list) or len(messages) < 2:
        return False
    valid_roles = {"system", "user", "assistant"}
    for m in messages:
        if not isinstance(m, dict):
            return False
        if m.get("role") not in valid_roles:
            return False
        if not m.get("content", "").strip():
            return False
    # Debe tener al menos un user y un assistant
    roles = {m["role"] for m in messages}
    return "user" in roles and "assistant" in roles


def load_and_validate(input_path: Path) -> list[dict]:
    """Carga JSONL y filtra ejemplos inválidos."""
    valid = []
    invalid_count = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                print(f"  [WARN] Línea {i}: JSON inválido, saltando.")
                invalid_count += 1
                continue
            if validate_message(msg):
                valid.append(msg)
            else:
                print(f"  [WARN] Línea {i}: formato de mensaje inválido, saltando.")
                invalid_count += 1
    print(f"  Cargados: {len(valid)} válidos, {invalid_count} descartados.")
    return valid


def split_dataset(
    data: list[dict], test_size: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """Split aleatorio estratificado."""
    random.seed(seed)
    shuffled = data.copy()
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - test_size))
    return shuffled[:split_idx], shuffled[split_idx:]


def write_jsonl(data: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Escrito: {path} ({len(data)} ejemplos)")


def main():
    parser = argparse.ArgumentParser(description="Preparar dataset para fine-tuning")
    parser.add_argument("--input", type=Path, required=True, help="JSONL de entrada")
    parser.add_argument("--output-train", type=Path, required=True, help="JSONL de train")
    parser.add_argument("--output-test", type=Path, required=True, help="JSONL de test")
    parser.add_argument("--test-size", type=float, default=0.1, help="Proporción de test (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="Seed para reproducibilidad")
    args = parser.parse_args()

    print(f"Cargando datos de {args.input}...")
    if not args.input.exists():
        print(f"ERROR: {args.input} no existe.")
        sys.exit(1)

    data = load_and_validate(args.input)
    if len(data) < 10:
        print(f"ERROR: Solo {len(data)} ejemplos válidos. Se necesitan al menos 10.")
        sys.exit(1)

    print(f"Splitting {len(data)} ejemplos (test_size={args.test_size})...")
    train, test = split_dataset(data, args.test_size, args.seed)

    write_jsonl(train, args.output_train)
    write_jsonl(test, args.output_test)
    print("Done.")


if __name__ == "__main__":
    main()
