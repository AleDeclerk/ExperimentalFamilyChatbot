"""
Fine-tune Falcon H1R 7B con QLoRA usando PEFT + TRL (compatible con Apple Silicon).

Uso:
    python scripts/finetune.py \
        --model-name tiiuae/Falcon-H1R-7B \
        --train-data data/processed/train.jsonl \
        --output-dir models/adapters/v1 \
        --epochs 3 \
        --batch-size 4 \
        --learning-rate 2e-4 \
        --lora-rank 16
"""

import argparse
import json
import platform
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer


def load_training_data(path: Path) -> Dataset:
    """Carga JSONL y convierte a Dataset de HuggingFace."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return Dataset.from_list(examples)


def format_chat(example: dict, tokenizer) -> str:
    """Aplica el chat template del tokenizer a los mensajes."""
    return tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )


def get_device_config():
    """Detecta el dispositivo óptimo y retorna configuración."""
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    elif torch.cuda.is_available():
        return "cuda", torch.float16
    else:
        return "cpu", torch.float32


def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning con PEFT + TRL")
    parser.add_argument("--model-name", type=str, default="tiiuae/Falcon-H1R-7B")
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--warmup-steps", type=int, default=50)
    args = parser.parse_args()

    device, dtype = get_device_config()
    is_apple = platform.processor() == "arm" or "Apple" in platform.platform()
    use_4bit = not is_apple  # BitsAndBytes 4-bit no soporta MPS

    print(f"Dispositivo: {device} | dtype: {dtype} | 4-bit: {use_4bit}")
    print(f"Cargando modelo {args.model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
    }

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"
    else:
        # Apple Silicon: cargar en float16 directo al MPS
        model_kwargs["device_map"] = {"": device}

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    model.gradient_checkpointing_enable()

    print(f"Configurando LoRA (rank={args.lora_rank}, alpha={args.lora_alpha})...")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"Cargando datos de {args.train_data}...")
    dataset = load_training_data(args.train_data)
    print(f"  {len(dataset)} ejemplos de entrenamiento.")

    dataset = dataset.map(
        lambda x: {"text": format_chat(x, tokenizer)},
        remove_columns=dataset.column_names,
    )

    sft_config = SFTConfig(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        fp16=(device == "cuda"),
        bf16=False,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        seed=42,
        dataloader_pin_memory=False,
        dataset_text_field="text",
        max_length=args.max_seq_length,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=sft_config,
    )

    print("Iniciando entrenamiento...")
    trainer.train()

    print(f"Guardando adapter en {args.output_dir}...")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print("Fine-tuning completado.")


if __name__ == "__main__":
    main()
