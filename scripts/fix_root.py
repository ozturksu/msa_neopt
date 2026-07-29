import re
from pathlib import Path

files = [
    'step02_train_pto.py',
    'step03_train_neopt.py',
    'step04_train_dbb.py',
    'step05_train_msa_neopt.py',
    'step06_evaluate.py',
]
for f in files:
    p = Path(f)
    txt = p.read_text()
    new = re.sub(
        r'ROOT\s*=\s*Path\(__file__\)\.parent(?!\.parent)',
        'ROOT      = Path(__file__).parent.parent',
        txt
    )
    p.write_text(new)
    root_line = [l for l in new.splitlines() if 'ROOT' in l and 'DIR' not in l][0].strip()
    print(f'{f}: {root_line}')
