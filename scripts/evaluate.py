"""
Evalúa el modelo fine-tuneado contra el test set.

Uso:
    python scripts/evaluate.py \
        --model-path tiiuae/Falcon-H1R-7B \
        --adapter-path models/adapters/v1 \
        --test-data data/processed/test.jsonl \
        --output eval/results_v1.json
"""

import argparse
import json
import platform
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_test_data(path: Path) -> list[dict]:
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def extract_prompt_and_reference(example: dict, tokenizer) -> tuple[str, str]:
    """Separa los mensajes en prompt (sin la última respuesta) y referencia."""
    messages = example["messages"]
    ref_msg = None
    prompt_msgs = []
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "assistant" and ref_msg is None:
            ref_msg = messages[i]["content"]
            prompt_msgs = messages[:i]
            break

    if ref_msg is None:
        return "", ""

    prompt = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True
    )
    return prompt, ref_msg


def main():
    parser = argparse.ArgumentParser(description="Evaluar modelo fine-tuneado")
    parser.add_argument("--model-path", type=str, default="tiiuae/Falcon-H1R-7B")
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-examples", type=int, default=None)
    args = parser.parse_args()

    device = get_device()
    print(f"Dispositivo: {device}")

    print(f"Cargando modelo desde {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map={"": device},
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    if args.adapter_path:
        print(f"Cargando adapter de {args.adapter_path}...")
        model = PeftModel.from_pretrained(model, str(args.adapter_path))

    model.eval()

    print(f"Cargando test data de {args.test_data}...")
    test_data = load_test_data(args.test_data)
    if args.max_examples:
        test_data = test_data[: args.max_examples]
    print(f"  {len(test_data)} ejemplos de evaluación.")

    results = []
    total_tokens = 0
    total_time = 0.0

    for i, example in enumerate(test_data):
        prompt, reference = extract_prompt_and_reference(example, tokenizer)
        if not prompt:
            continue

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        elapsed = time.time() - start

        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        gen_tokens = outputs[0].shape[0] - inputs["input_ids"].shape[1]
        total_tokens += gen_tokens
        total_time += elapsed

        results.append({
            "index": i,
            "reference": reference,
            "generated": generated,
            "tokens": gen_tokens,
            "time_s": round(elapsed, 2),
            "tok_per_s": round(gen_tokens / elapsed, 1) if elapsed > 0 else 0,
        })

        if (i + 1) % 5 == 0:
            print(f"  [{i + 1}/{len(test_data)}] avg {total_tokens / total_time:.1f} tok/s")

    summary = {
        "model": args.model_path,
        "adapter": str(args.adapter_path) if args.adapter_path else None,
        "device": device,
        "num_examples": len(results),
        "total_tokens": total_tokens,
        "total_time_s": round(total_time, 2),
        "avg_tok_per_s": round(total_tokens / total_time, 1) if total_time > 0 else 0,
        "results": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Resultados guardados en {args.output}")
    print(f"Promedio: {summary['avg_tok_per_s']} tok/s sobre {len(results)} ejemplos.")


if __name__ == "__main__":
    main()
