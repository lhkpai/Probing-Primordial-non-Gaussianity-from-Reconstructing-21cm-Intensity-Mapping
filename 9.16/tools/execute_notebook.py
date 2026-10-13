"""Execute the edited notebook in a clean kernel and save actual outputs."""
from pathlib import Path
import time
import nbformat
from nbclient import NotebookClient

path = Path(__file__).resolve().parents[1] / '9.16-interferometer-telescopes.ipynb'
notebook = nbformat.read(path,as_version=4)
start = time.monotonic()
def report(cell,cell_index,**kwargs):
    print(f'cell {cell_index} finished; elapsed {time.monotonic()-start:.1f}s',flush=True)
client = NotebookClient(notebook,timeout=7200,kernel_name='python3',
                        resources={'metadata':{'path':str(path.parent)}},
                        on_cell_executed=report)
client.execute()
nbformat.write(notebook,path)
print(f'Notebook execution complete: {time.monotonic()-start:.1f}s',flush=True)
