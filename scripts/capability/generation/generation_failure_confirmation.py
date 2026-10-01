#!/usr/bin/env python3
"""Small native generation confirmation; stop at residue cap without alphabet constraints."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.capability.generation import generation_failure as gf
from src.capability.generation import generation_evidence as ge
from src.capability.generation import conditioned_generation as cg
from src.capability.generation import unconditional_generation as ug
from src.capability.core.io import write_json, sha256_file


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', required=True, choices=['prollama-stage-1','prollama','progen3-112m','progen3-3b'])
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--attempts', type=int, default=64)
    p.add_argument('--residue-budget', type=int, default=400)
    p.add_argument('--token-safety-cap', type=int, default=1024)
    p.add_argument('--seed', type=int, default=20260923)
    p.add_argument('--hmmscan', type=Path)
    p.add_argument('--pfam-hmm', type=Path)
    p.add_argument('--ledger', type=Path, help='decompose an existing ledger instead of generating')
    a=p.parse_args()
    if min(a.attempts,a.residue_budget,a.token_safety_cap)<1: p.error('budgets must be positive')
    a.out.mkdir(parents=True,exist_ok=True)
    if a.ledger:
        rows=[]
        for r in ug.read_attempts(a.ledger):
            if r.get('arm') != a.arm: raise ValueError('ledger arm mismatch')
            d=gf.decompose(r['raw_continuation'],a.arm)
            d.update({k:r[k] for k in ['id','any_profile_hit'] if k in r})
            rows.append(d)
        provenance={'source_ledger':str(a.ledger),'source_sha256':sha256_file(a.ledger)}
    else:
        rows,provenance=generate(a)
    if bool(a.hmmscan)!=bool(a.pfam_hmm): p.error('both Pfam paths required')
    if a.hmmscan:
        hits,receipt=cg.annotate({str(i):r['sequence'] for i,r in enumerate(rows)}, tool=SimpleNamespace(hmmscan=a.hmmscan), database=SimpleNamespace(path=a.pfam_hmm),workspace=a.out/'pfam',threads=4,shards=2,label='generation_failure_confirmation')
        for i,r in enumerate(rows):
            r['pfam_families']=ge._families(hits,str(i)); r['any_profile_hit']=bool(r['pfam_families'])
        provenance['pfam_receipt']=receipt
    ge.write_immutable(a.out/'attempts.jsonl',ge.jsonl_bytes(rows))
    write_json(a.out/'generation_failure_summary.json',{'arm':a.arm,'policy':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},'provenance':provenance,**gf.summarize(rows)})


def generate(a, *, loaded=None):
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList
    progen=a.arm.startswith('progen3')
    if progen:
        from src.capability.models.progen3 import load_progen3
        from src.capability.core.arms import arm_spec
        from src.capability.generation.progen3_generation import make_generator,generation_self_check
        pg=load_progen3(arm_spec(a.arm).path,device=a.device,dtype=torch.bfloat16)
        check=generation_self_check(pg)
        gen=make_generator(pg)
        model=pg.model; tok=gen.batch_preparer.tokenizer
        encoded=gen.batch_preparer.get_generation_kwargs('',False)
        eos=tok.token_to_id('<eos>'); pad=tok.padding['pad_id']
        def decode(ids): return tok.decode(ids,skip_special_tokens=False)
    else:
        model,tok=ug.load_generator(a.arm,device=a.device) if loaded is None else loaded
        encoded=tok(ug.prompt_for(a.arm),return_tensors='pt')
        eos=tok.eos_token_id; pad=tok.pad_token_id if tok.pad_token_id is not None else eos
        def decode(ids): return tok.decode(ids,skip_special_tokens=False)
        check={'strict_loader':'load_rung'}
    encoded={k:v.to(a.device) for k,v in encoded.items()}
    prompt_len=encoded['input_ids'].shape[1]
    rows=[]
    class ResidueStop(StoppingCriteria):
        def __init__(self): self.trace={}
        def __call__(self,input_ids,scores,**kwargs):
            stopped=[]
            for i,ids in enumerate(input_ids):
                tail=ids[prompt_len:].tolist()
                d=gf.decompose(decode(tail),a.arm,residue_budget=a.residue_budget)
                reason=('native_terminal' if d['native_terminal_observed'] else
                        'model_eos' if tail[-1]==eos else
                        'residue_budget' if d['residue_length']>=a.residue_budget else None)
                if reason and i not in self.trace:self.trace[i]=(len(tail),reason)
                stopped.append(i in self.trace)
            return torch.tensor(stopped,device=input_ids.device)
    for start in range(0,a.attempts,8):
        size=min(8,a.attempts-start); stop=ResidueStop()
        torch.manual_seed(a.seed+start//8)
        batch={k:v.repeat(size,1) for k,v in encoded.items()}
        with torch.no_grad():
            output=model.generate(**batch,do_sample=True,temperature=.85,top_p=.95,top_k=50,repetition_penalty=1.,max_new_tokens=a.token_safety_cap,min_new_tokens=2 if progen else 0,pad_token_id=pad,eos_token_id=eos,use_cache=True,stopping_criteria=StoppingCriteriaList([stop]))
        if hasattr(output,'sequences'): output=output.sequences
        for i,ids in enumerate(output):
            tail=ids[prompt_len:].tolist(); length,reason=stop.trace.get(i,(len(tail),'token_safety_cap'))
            tail=tail[:length]
            r=gf.decompose(decode(tail),a.arm,residue_budget=a.residue_budget,stop_reason=reason)
            r.update(arm=a.arm,attempt_index=start+i,generated_token_ids=tail,generated_tokens=len(tail))
            rows.append(r)
        print(f'{a.arm}: {len(rows)}/{a.attempts}',flush=True)
        write_json(a.out/'progress.json',{'attempts':len(rows)})
    return rows,{'self_check':check,'torch_version':torch.__version__,'runner_sha256':sha256_file(Path(__file__)),'module_sha256':sha256_file(Path(gf.__file__)),'sampling':'native distribution, no forced alphabet; independent draws, batch8; cap token overshoot retained'}

if __name__=='__main__':main()
