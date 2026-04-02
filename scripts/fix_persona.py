"""
Reescribe todos los datos de training para que el asistente mantenga
su postura como asistente virtual, nunca como familiar.

Cambios:
1. System prompt actualizado: es asistente virtual, no familiar
2. Respuestas: elimina expresiones de parentesco ("يا ولدي", "يا حبيبتي")
   y las reemplaza por expresiones de asistente amable
3. Mantiene el tono emiratí cálido pero profesional
"""

import json
import re
from pathlib import Path

NEW_SYSTEM_PROMPT = (
    "أنت مساعد افتراضي ودود متخصص في الشؤون العائلية، تتحدث باللهجة الإماراتية. "
    "تقدم النصائح والمساعدة لأفراد العائلة في حياتهم اليومية بأسلوب دافئ ومحترم. "
    "أنت دائماً مساعد وليس فرداً من العائلة. لا تتقمص دور أم أو أب أو جد أو أي قريب. "
    "استخدم التعبيرات الإماراتية الشائعة وحافظ على أسلوب مهذب ومهني."
)

# Replacements: familiar terms → assistant-friendly terms
REPLACEMENTS = [
    # Remove "ya habibti/habibi" (my love) → "ya alghali/ya alghalia" (dear)
    ("يا حبيبتي", "يا الغالية"),
    ("يا حبيبي", "يا الغالي"),
    ("حبيبتي", "يا الغالية"),
    ("حبيبي", "يا الغالي"),
    # Remove parental terms
    ("يا ولدي", ""),
    ("يا بنتي", ""),
    ("يا عمري", ""),
    ("يا قلبي", ""),
    ("حبيب بابا", ""),
    ("حبيب قلبي", ""),
    # Remove "taal ya" (come, child)
    ("تعالي يا بنتي", "تعالي"),
    ("تعال يا ولدي", "تعال"),
    # First person as family → as assistant
    ("بنسوي لك", "تقدرين تسوين"),
    ("بعلمك", "بقولك الطريقة"),
    ("بسوي لك", "تقدر تسوي"),
    # "anا" as parent → neutral
    ("أنا بوديها", "يفضل توديها"),
    ("أنا بوديه", "يفضل توديه"),
    ("بأسوي لك", "تقدر تسوي"),
]

# Patterns that indicate the bot is acting as a family member
FAMILY_PATTERNS = [
    r"أبوك يحبك",
    r"أمك تحبك",
    r"بابا يحبك",
    r"يمه سويت لك",
    r"أبوك وياك",
]


def fix_response(text: str) -> str:
    """Fix assistant response to maintain virtual assistant persona."""
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)

    # Clean up double spaces from removals
    text = re.sub(r"  +", " ", text)
    text = text.strip()
    # Remove leading comma or space
    text = re.sub(r"^[،, ]+", "", text)
    return text


def fix_example(example: dict) -> dict:
    """Fix a single training example."""
    messages = example["messages"]
    fixed = []
    for msg in messages:
        if msg["role"] == "system":
            fixed.append({"role": "system", "content": NEW_SYSTEM_PROMPT})
        elif msg["role"] == "assistant":
            fixed.append({"role": "assistant", "content": fix_response(msg["content"])})
        else:
            fixed.append(msg)
    return {"messages": fixed}


def main():
    input_files = [
        Path("data/raw/conversations.jsonl"),
        Path("data/raw/conversations_1500.jsonl"),
    ]
    output_path = Path("data/raw/all_conversations_v3.jsonl")

    all_examples = []
    seen = set()

    for input_path in input_files:
        if not input_path.exists():
            print(f"Skipping {input_path} (not found)")
            continue
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                example = json.loads(line)
                fixed = fix_example(example)

                # Dedup by user message
                user_msgs = tuple(m["content"] for m in fixed["messages"] if m["role"] == "user")
                if user_msgs not in seen:
                    seen.add(user_msgs)
                    all_examples.append(fixed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Total: {len(all_examples)} ejemplos (deduplicados) en {output_path}")

    # Filter out examples where the user addresses the bot as a family member
    # and the bot responds as that family member
    filtered = []
    family_address = ["يا يدي", "يا يمه", "يا بابا", "يا يدتي", "يا أبوي"]
    first_person_family = [
        "وأنا أكثر مشتاق", "بابا يحبك", "أبوك يحبك", "يدتك",
        "أنا أكثر مشتاق", "حبيب بابا", "يا عيالي",
        "بعد عمري", "يا نور عيني",
    ]

    for ex in all_examples:
        # Check if any assistant message acts as family
        dominated_by_family = False
        for msg in ex["messages"]:
            if msg["role"] == "assistant":
                if any(term in msg["content"] for term in first_person_family):
                    dominated_by_family = True
                    break
        if not dominated_by_family:
            filtered.append(ex)

    removed = len(all_examples) - len(filtered)
    print(f"Eliminados {removed} ejemplos donde el bot actúa como familiar")

    # Rewrite output
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in filtered:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Final: {len(filtered)} ejemplos en {output_path}")


if __name__ == "__main__":
    main()
