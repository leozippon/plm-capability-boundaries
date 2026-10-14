"""Single source for concise plot labels and the public checkpoint-name key."""
from pathlib import Path
import csv
import io

if __package__:
    from . import publication_paths
else:
    import publication_paths

PATH = Path(__file__).resolve()
# Canonical IDs are retained in every plotted-source table; aliases change only text.
CHECKPOINTS = {
    'protgpt2': ('ProtGPT2', 'ProtGPT2'),
    'progen2-small': ('PG2-S', 'ProGen2 small'),
    'progen2-base': ('PG2-B', 'ProGen2 base'),
    'progen2-medium': ('PG2-M', 'ProGen2 medium'),
    'progen2-large': ('PG2-L', 'ProGen2 large'),
    'progen2-xlarge': ('PG2-XL', 'ProGen2 extra-large'),
    'progen3-112m': ('PG3-112M', 'ProGen3 112 million parameters'),
    'progen3-3b': ('PG3-3B', 'ProGen3 3 billion parameters'),
    'protgpt3-1.3b': ('PGPT3-1.3B', 'ProtGPT3 1.3 billion parameters'),
    'rita-xl': ('RITA-XL', 'RITA extra-large'),
    'galactica-125m': ('Gal-125M', 'Galactica 125 million parameters'),
    'galactica-1.3b': ('Gal-1.3B', 'Galactica 1.3 billion parameters'),
    'galactica-6.7b': ('Gal-6.7B', 'Galactica 6.7 billion parameters'),
    'galactica-30b': ('Gal-30B', 'Galactica 30 billion parameters'),
    'llama-2-7b': ('L2-7B', 'Llama 2, 7 billion parameters'),
    'prollama-stage-1': ('ProL-S1', 'ProLLaMA stage 1 (protein-adapted)'),
    'prollama': ('ProL-S2', 'ProLLaMA stage 2 (instruction-tuned)'),
    'proteinglm-7b-clm': ('PGLM-7B', 'ProteinGLM 7 billion parameters, causal language model'),
    'instructprotein': ('InstProt', 'InstructProtein'),
    'gpt2': ('GPT2-S', 'GPT-2 small (124 million parameters)'),
    'gpt2-medium': ('GPT2-M', 'GPT-2 medium (355 million parameters)'),
    'gpt2-large': ('GPT2-L', 'GPT-2 large (774 million parameters)'),
    'gpt2-xl': ('GPT2-XL', 'GPT-2 extra-large (1.5 billion parameters)'),
    'dialogpt-small': ('Dialo-S', 'DialoGPT small'),
    'bygpt5-small-en': ('ByG5-S', 'ByGPT5 small, English'),
    'bygpt5-base-en': ('ByG5-B', 'ByGPT5 base, English'),
    'bygpt5-medium-en': ('ByG5-M', 'ByGPT5 medium, English'),
    'qwen2.5-0.5b': ('Q2.5-0.5B', 'Qwen 2.5, 0.5 billion parameters, base'),
    'qwen2.5-0.5b-instruct': ('Q2.5-0.5B-I', 'Qwen 2.5, 0.5 billion parameters, instruction-tuned'),
    'qwen2.5-7b': ('Q2.5-7B', 'Qwen 2.5, 7 billion parameters, base'),
    'qwen2.5-32b': ('Q2.5-32B', 'Qwen 2.5, 32 billion parameters, base'),
    'qwen3-8b-base': ('Q3-8B-B', 'Qwen 3, 8 billion parameters, base'),
    'llama-3.2-3b': ('L3.2-3B', 'Llama 3.2, 3 billion parameters, base'),
    'zymctrl': ('ZymCTRL', 'ZymCTRL enzyme-conditioned protein generator'),
}


def label(arm):
    """Fail explicitly for unknown checkpoint IDs; support retained generation cells."""
    checkpoint, *condition = arm.split('__')
    short = CHECKPOINTS[checkpoint][0]
    return short + ('†' if any('requested' in item for item in condition) else '')


def key_csv():
    output = io.StringIO(newline='')
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(['checkpoint_id', 'short_label', 'full_name'])
    writer.writerows((arm, short, full) for arm, (short, full) in CHECKPOINTS.items())
    return output.getvalue()


def key_tex():
    """Standard tabular only: no extra manuscript package dependencies."""
    rows = '\n'.join(f'{short} & {full} \\\\' for short, full in CHECKPOINTS.values())
    return ('% Generated from checkpoint_labels.py; do not edit independently.\n'
            '\\begin{table}[!htbp]\n\\centering\n\\small\n'
            '\\caption{Checkpoint-name key for all quantitative figures. '
            'S/B/M/L/XL denote small/base/medium/large/extra-large; M and B after a number denote million and billion parameters. '
            'S1/S2 preserve adaptation stage and I denotes instruction tuning. '
            '$\\dagger$ marks a requested-condition generation cell, not a different checkpoint; '
            'unmarked cells use the unconditioned procedure. Mismatched conditions are labelled explicitly.}\n'
            '\\label{tab:checkpoint-key}\n'
            '\\begin{tabular}{lp{0.73\\textwidth}}\n\\hline\nShort label & Full checkpoint identity \\\\\n\\hline\n'
            + rows + '\n\\hline\n\\end{tabular}\n\\end{table}\n')


def write_key(repo):
    csv_path = repo / 'scripts/capability/reporting/figure_data/checkpoint-name-key.csv'
    tex_path = repo / publication_paths.CHECKPOINT_KEY
    for path, content in [(csv_path, key_csv()), (tex_path, key_tex())]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return [csv_path, tex_path]
