#!/usr/bin/env python3
"""Stage4 English->Korean document-context translation. Inference only.

Uses Stage3-trained-and-continued PEFT adapter plus exact Stage4 prompt and
conditional repeat recovery rule. No references or test labels are accessed.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any

from prompts import prompt

HERE = Path(__file__).resolve().parent


def abnormal_reasons(text: str, hit_cap: bool) -> list[str]:
    out: list[str] = []
    if not text.strip():
        out.append('empty')
    if hit_cap:
        out.append('token_cap')
    if '\ufffd' in text:
        out.append('replacement_character')
    words = re.findall(r'\S+', text)
    counts = Counter(tuple(words[i:i+6]) for i in range(max(0, len(words) - 5)))
    if counts and max(counts.values()) >= 3:
        out.append('repeated_six_word_phrase')
    return out


class Stage4Translator:
    def __init__(self, adapter_dir: str | Path = HERE / 'adapter',
                 glossary_path: str | Path = HERE / 'strict_glossary.json',
                 max_input_tokens: int = 1536, max_new_tokens: int = 512):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftConfig, PeftModel
        self.adapter_dir = Path(adapter_dir).resolve()
        if not (self.adapter_dir/'adapter_config.json').is_file():
            raise FileNotFoundError('adapter_config.json missing in the selected LoRA adapter')
        if not (self.adapter_dir/'adapter_model.safetensors').is_file():
            raise FileNotFoundError('adapter_model.safetensors missing in the selected LoRA adapter')
        if not torch.cuda.is_available():
            raise RuntimeError('A CUDA GPU is required for this validated bf16 configuration.')
        self.glossary = json.loads(Path(glossary_path).read_text(encoding='utf-8'))
        if not isinstance(self.glossary, list):
            raise ValueError('The strict glossary must be a JSON list.')
        cfg = PeftConfig.from_pretrained(str(self.adapter_dir))
        base = cfg.base_model_name_or_path
        if not base:
            raise ValueError('Missing base_model_name_or_path in adapter_config.json')
        revision = getattr(cfg, 'revision', None)
        kw = {'trust_remote_code': True}
        if revision:
            kw['revision'] = revision
        tok_src = str(self.adapter_dir) if (self.adapter_dir/'tokenizer_config.json').is_file() else base
        self.tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None:
            raise ValueError('Tokenizer has no pad or EOS token')
        self.model = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map={'': 'cuda:0'}, **kw),
            str(self.adapter_dir), is_trainable=False).eval()
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.base_model = base

    def build_prompt(self, current_en: str, prev_en: str = '') -> str:
        return prompt({'current_en': current_en, 'prev_en': prev_en}, 'context', self.glossary)

    def _generate(self, current_en: str, prev_en: str, retry: bool) -> dict[str, Any]:
        import torch
        from transformers import GenerationConfig
        content = self.build_prompt(current_en, prev_en)
        x = self.tokenizer.apply_chat_template(
            [{'role':'user','content':content}], tokenize=True,
            add_generation_prompt=True, return_tensors='pt', return_dict=True).to(self.model.device)
        n = x['input_ids'].shape[-1]
        if n > self.max_input_tokens:
            raise ValueError(f'Prompt length {n} > max_input_tokens={self.max_input_tokens}; refusing silent truncation')
        eos = self.model.generation_config.eos_token_id
        eos = list(eos) if isinstance(eos, (list, tuple)) else ([] if eos is None else [eos])
        if self.tokenizer.eos_token_id is not None and self.tokenizer.eos_token_id not in eos:
            eos.append(self.tokenizer.eos_token_id)
        if not eos:
            raise ValueError('No EOS token ids configured')
        conf = GenerationConfig(
            do_sample=False, num_beams=1, max_new_tokens=self.max_new_tokens,
            use_cache=True, eos_token_id=eos, pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            repetition_penalty=1.05 if retry else 1.0,
            no_repeat_ngram_size=12 if retry else 0)
        with torch.inference_mode():
            y = self.model.generate(**x, generation_config=conf)
        new = y[0, n:].tolist()
        end = next((i for i, t in enumerate(new) if t in eos), None)
        result = new if end is None else new[:end]
        text = self.tokenizer.decode(result, skip_special_tokens=True).strip()
        cap = end is None and len(new) >= self.max_new_tokens
        return {'translation': text, 'hit_cap': cap,
                'generated_token_count': len(result), 'retry_used': retry,
                'abnormal_reasons': abnormal_reasons(text, cap)}

    def translate(self, current_en: str, prev_en: str = '') -> dict[str, Any]:
        if not current_en.strip():
            raise ValueError('current_en must not be empty')
        first = self._generate(current_en, prev_en, retry=False)
        if not first['abnormal_reasons']:
            return first
        again = self._generate(current_en, prev_en, retry=True)
        again['initial_abnormal_reasons'] = first['abnormal_reasons']
        return again


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--adapter', default=str(HERE/'adapter'))
    p.add_argument('--glossary', default=str(HERE/'strict_glossary.json'))
    p.add_argument('--text', required=True, help='CURRENT English source sentence')
    p.add_argument('--previous', default='', help='Prior English context, up to two sentences')
    p.add_argument('--json', action='store_true', help='Include metadata on retry and truncation')
    a = p.parse_args()
    output = Stage4Translator(a.adapter, a.glossary).translate(a.text, a.previous)
    print(json.dumps(output, ensure_ascii=False, indent=2) if a.json else output['translation'])


if __name__ == '__main__':
    main()
