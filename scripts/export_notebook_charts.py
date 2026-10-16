"""Export the final comparison charts saved in the notebook (standard library only)."""
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'SpamClassifiers.ipynb'
DESTINATION = ROOT / 'assets' / 'charts'


def main():
    notebook = json.loads(NOTEBOOK.read_text())
    chart_cells = [
        cell for cell in notebook['cells']
        if cell['cell_type'] == 'code'
        and 'Final family comparison' in ''.join(cell.get('source', []))
    ]
    if len(chart_cells) != 1:
        raise SystemExit('Expected one final comparison chart cell in the notebook.')
    images = [
        output['data']['image/png']
        for output in chart_cells[0].get('outputs', [])
        if 'image/png' in output.get('data', {})
    ]
    if len(images) != 2:
        raise SystemExit('Run and save the notebook first; expected two saved PNG charts.')
    decoded = [base64.b64decode(''.join(''.join(image).split()), validate=True) for image in images]
    if not all(image.startswith(b'\x89PNG\r\n\x1a\n') for image in decoded):
        raise SystemExit('A saved chart is not a PNG image.')
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for name, data in zip(('model-comparison.png', 'logistic-confusion-matrix.png'), decoded):
        path = DESTINATION / name
        path.write_bytes(data)
        print(path.relative_to(ROOT))


if __name__ == '__main__':
    main()
