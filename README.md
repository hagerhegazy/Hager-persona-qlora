# Hager Persona — QLoRA Fine-Tuning

A personal AI chatbot fine-tuned to answer questions about my education, projects, and experience — trained using QLoRA on top of Qwen2.5-3B-Instruct.

## Overview

This project fine-tunes [Qwen2.5-3B-Instruct](https://huggingface.co/unsloth/Qwen2.5-3B-Instruct) using **QLoRA** (4-bit quantized LoRA) via [Unsloth](https://github.com/unslothai/unsloth) and [TRL](https://github.com/huggingface/trl), on a custom Q&A dataset built from my CV — covering my education, projects, work experience, and skills, in both English and Arabic.

The result is a small conversational model that can answer questions like *"Tell me about your NASA project"* or *"ما هي مهاراتك؟"* in a consistent, first-person voice.

## Dataset

- 120 instruction-style Q&A pairs, formatted as chat messages (`{"role": "user"/"assistant", "content": ...}`)
- Bilingual: English and Arabic
- Topics: education, GPA/ranking, technical skills, NASA Exoplanet Detection project, Shifaa graduation project, work experience, certifications, career goals

## Training details

- **Base model:** Qwen2.5-3B-Instruct (4-bit quantized)
- **Method:** QLoRA (r=16, alpha=16) via Unsloth's `FastLanguageModel`
- **Target modules:** q/k/v/o/gate/up/down projections
- **Framework:** TRL `SFTTrainer`
- **Epochs:** 5
- **Hardware:** Kaggle GPU notebook

## Files

- `qlora-finetune.ipynb` — full training notebook: setup, dataset prep, LoRA config, training, evaluation, and inference testing

## Usage

The merged fine-tuned model is available on the Hugging Face Hub: [`Hager290/hager-persona`](https://huggingface.co/Hager290/hager-persona)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Hager290/hager-persona")
tokenizer = AutoTokenizer.from_pretrained("Hager290/hager-persona")
```

## Acknowledgments

Built with [Unsloth](https://github.com/unslothai/unsloth) for efficient QLoRA fine-tuning.
