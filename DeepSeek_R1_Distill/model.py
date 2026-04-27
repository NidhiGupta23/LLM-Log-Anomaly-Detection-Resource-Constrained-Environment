import re
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import Config
from memory import get_cpu_memory_mb, get_system_memory_percent

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a BGL (Blue Gene/L) supercomputer log anomaly classifier.

You will receive a structured log entry with the following fields:
  TYPE      – high-level event type (e.g. RAS)
  COMPONENT – subsystem that emitted the event (e.g. KERNEL, APP)
  LEVEL     – severity (e.g. FATAL, INFO, WARNING)
  CONTENT   – free-text description of the event

Classify the entry as:
  0 = NORMAL
  1 = ABNORMAL

CRITICAL rules:
- Do NOT classify based on LEVEL alone.
- FATAL does NOT automatically mean abnormal — many FATAL lines are
  diagnostic register dumps that are part of normal error recovery.
- Focus on CONTENT: what actually happened?

ABNORMAL indicators in CONTENT:
- Kernel or process termination  ("kernel terminated", "rts panic",
  "stopping execution", "job terminated", "killed by signal")
- ciod socket / read failures    ("ciod: error reading message",
  "ciod: read error", "ciod: socket error", "ciod: failed to connect")
- Network packet errors          ("error receiving packet on tree network",
  "link has been severed", "connection timed out", "connection reset by peer")
- Storage / mount failures       ("Lustre mount FAILED", "data TLB error interrupt",
  "data storage interrupt", "I/O error")
- Illegal instruction / machine check interrupt

NORMAL indicators in CONTENT:
- Register dump lines            ("rts internal error", "instruction address",
  "machine state register", "exception syndrome register")
- Self-corrected hardware errors ("detected and corrected", "cache parity error
  corrected", "ddr error(s) detected and corrected", "CE sym")
- Transient recoverable failures ("NFS mount failed … retrying",
  "suppressing further interrupts")

EXAMPLES:
  TYPE=RAS COMPONENT=KERNEL LEVEL=FATAL CONTENT=rts internal error          → 0
  TYPE=RAS COMPONENT=KERNEL LEVEL=FATAL CONTENT=instruction address: …      → 0
  TYPE=RAS COMPONENT=KERNEL LEVEL=INFO  CONTENT=ddr error(s) detected …    → 0
  TYPE=RAS COMPONENT=KERNEL LEVEL=INFO  CONTENT=NFS Mount failed, retrying  → 0
  TYPE=RAS COMPONENT=KERNEL LEVEL=FATAL CONTENT=rts: kernel terminated      → 1
  TYPE=RAS COMPONENT=KERNEL LEVEL=FATAL CONTENT=rts panic! stopping exec    → 1
  TYPE=RAS COMPONENT=APP    LEVEL=FATAL CONTENT=Link has been severed        → 1
  TYPE=RAS COMPONENT=APP    LEVEL=FATAL CONTENT=Connection timed out         → 1
  TYPE=RAS COMPONENT=APP    LEVEL=FATAL CONTENT=ciod: error reading message  → 1
  TYPE=RAS COMPONENT=KERNEL LEVEL=FATAL CONTENT=Lustre mount FAILED          → 1
  TYPE=RAS COMPONENT=KERNEL LEVEL=FATAL CONTENT=Error receiving packet on
    tree network (expecting type …)                                          → 1

Return exactly one digit: 0 or 1. No explanation.
"""


def build_prompt(entry: "BGLEntry") -> str:  # type: ignore[name-defined]
    return (
        f"{_SYSTEM_PROMPT}\n"
        f"TYPE={entry.type} "
        f"COMPONENT={entry.component} "
        f"LEVEL={entry.level} "
        f"CONTENT={entry.content}\n"
        f"OUTPUT:"
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(config: Config):
    """Download (or load from cache) the tokenizer and model, return both."""
    print("Loading tokenizer and model...")
    cpu_before = get_cpu_memory_mb()

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    cpu_after = get_cpu_memory_mb()
    print(f"  Model device              : {next(model.parameters()).device}")
    print(f"  CPU memory after loading  : {cpu_after:.1f} MB")
    print(f"  CPU memory model increase : {cpu_after - cpu_before:.1f} MB")
    print(f"  System RAM usage          : {get_system_memory_percent():.1f}%")

    return tokenizer, model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _parse_prediction(text: str) -> int:
    """Extract the first 0 or 1 from generated text; default to 0."""
    text = text.strip().replace("<think>", "").replace("</think>", "")
    match = re.search(r"[01]", text)
    return int(match.group()) if match else 0


def llm_classify_batch(
    log_lines: List[str],
    tokenizer,
    model,
    config: Config,
) -> List[int]:
    """Run a single batch of log lines through the LLM and return predictions."""
    prompts = [build_prompt(log) for log in log_lines]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.max_length,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_lengths = inputs["attention_mask"].sum(dim=1)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    predictions = []
    for output, input_len in zip(outputs, input_lengths):
        generated_tokens = output[input_len:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        predictions.append(_parse_prediction(generated_text))

    return predictions
