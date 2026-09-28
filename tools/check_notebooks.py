"""Execute both notebooks in isolated directories using this Python interpreter.

--write retains executed notebook outputs for public static previews. Each fresh
kernel uses the embedded solver: no local source checkout or GitHub URL needed.
"""
import argparse
import base64
import json
import os
from pathlib import Path
import sys
import tempfile
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--write', action='store_true')
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix='fronttrack-check-') as scratch:
    scratch = Path(scratch)
    kernel = scratch/'kernels'/'fronttrack-check'
    kernel.mkdir(parents=True)
    (kernel/'kernel.json').write_text(json.dumps({
        'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'Fronttrack check', 'language': 'python',
        'env': {'MPLCONFIGDIR': str(scratch/'mpl')},
    }))
    # Only the bundled notebooks run in isolation. 03/04 need the checkout and are
    # executed by tools/build_theory_notebooks.py --execute instead.
    for path in sorted((root/'notebooks').glob('0[12]_*.ipynb')):
        work = scratch/path.stem
        work.mkdir()
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        # Exercise callbacks too, including changes of time, palette and example.
        probe = nbformat.v4.new_code_cell('''
if 'selector' in globals():
    for name in experiments:
        selector.value = name
    panel = viewer_box.children[0]
else:
    panel = viewer
panel.children[0].children[1].value = 37
panel.children[1].children[0].value = 'cividis'
panel.children[1].children[1].value = False
assert panel.children[0].children[1].value == 37
# Leave the saved controls at their initial presentation.
panel.children[0].children[1].value = 0
panel.children[1].children[0].value = 'viridis'
panel.children[1].children[1].value = True
if 'selector' in globals():
    selector.value = next(iter(experiments))
''')
        nb.cells.append(probe)
        manager = KernelManager(kernel_name='fronttrack-check',
                                kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel.parent)]))
        try:
            NotebookClient(nb, km=manager, timeout=180, resources={'metadata': {'path': str(work)}}).execute()
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
        nb.cells.pop()
        nbformat.validate(nb)
        if args.write:
            nbformat.write(nb, path)
            if path.name.startswith('02_'):
                png = next(output['data']['image/png']
                           for cell in nb.cells for output in cell.get('outputs', [])
                           if 'image/png' in output.get('data', {}))
                (root/'docs').mkdir(exist_ok=True)
                (root/'docs/examples.png').write_bytes(base64.b64decode(png))
        print('PASS:', path.name, flush=True)
