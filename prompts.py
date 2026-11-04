"""Prompt/policy functions copied from the prepared data; no new re-alignment."""
import json, re, unicodedata, hashlib
from pathlib import Path

BASE = ('Translate the English academic text into natural Korean. '
        'Preserve technical proper names and any Markdown formatting.')
SEED = 20261103

def read(p):
    with Path(p).open(encoding='utf-8-sig') as f:
        return [json.loads(l) for l in f if l.strip()]

def dump(p, value):
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2),encoding='utf-8')

def write(p, rows):
    with Path(p).open('w',encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stable(xs, seed):
    return sorted(xs,key=lambda x:hashlib.sha256(f'{seed}|{x}'.encode()).digest())
def normalized(s): return ''.join(unicodedata.normalize('NFC',s).split()).casefold()

def terms(source, gloss):
    """Conservative subset of the existing policy; never use references to construct rules."""
    hits=[]; seen=set()
    for e in gloss:
        if not e.get('auto_enforce', True) or e.get('context_sensitive',False): continue
        variants=[(e['term'],e['action']),*((v,e['action']) for v in e.get('variants',[]))]
        if e.get('abbreviation'): variants.append((e['abbreviation'],'KEEP'))
        for phrase, action in variants:
            if not phrase or re.search('[가-힣]',phrase): continue
            sensitive=bool(re.fullmatch(r'[A-Z][A-Z0-9-]{1,9}',phrase))
            pattern=re.compile(r'(?<![A-Za-z0-9_])'+re.escape(phrase)+r'(?![A-Za-z0-9_])',0 if sensitive else re.I)
            for m in pattern.finditer(source):
                key=(m.group().casefold(),action)
                if key in seen: continue
                seen.add(key)
                allowed=list(dict.fromkeys(v for v in [e.get('translation'),*e.get('translation_variants',[])] if v))
                if action=='TRANSLATE' and not allowed: continue
                hits.append({'source':m.group(),'action':action,'allowed':allowed})
    return sorted(hits,key=lambda h:(h['source'].casefold(),h['action']))

def follows(ref,h):
    if h['action']=='KEEP':
        return re.search(r'(?<![A-Za-z0-9_])'+re.escape(h['source'])+r'(?![A-Za-z0-9_])',ref) is not None
    return any(normalized(v) in normalized(ref) for v in h['allowed'])

def prompt(row, mode, gloss):
    rules=[]
    for h in terms(row['current_en'],gloss):
        if h['action']=='KEEP': rules.append(f"- Keep '{h['source']}' exactly in English.")
        else: rules.append(f"- Translate '{h['source']}' using a suitable Korean term from: {', '.join(h['allowed'])}.")
    instruction=BASE+ ('\nFollow these terminology rules:\n'+'\n'.join(rules) if rules else '')
    if mode=='plain' or not row['prev_en'].strip():
        return instruction+'\n\n'+row['current_en']
    return ('[Background Information]\n'+row['prev_en']+'\n\n'+instruction+
            '\nUse the background only to resolve meaning. Translate only [Source Text]; '
            'do not translate or summarize the background. Output only the translation.\n'
            '[Source Text]\n'+row['current_en'])

