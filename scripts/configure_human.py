"""Bind external human checkpoints from an explicit runtime config; never download weights."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'human'
LINKS = {'vendor/4danyone/models', 'vendor/4danyone/third_party/GVHMR/inputs/checkpoints',
         'vendor/metric_weights'}


def configure(config, *, apply=False):
    value = json.loads(config.read_text())
    if set(value['links']) != LINKS:
        raise ValueError('Expected exactly the three documented external checkpoint directories')
    missing = [p for p in [*value['pythons'].values(), *value['resources'].values(), *value['links'].values()]
               if not Path(p).is_absolute() or not Path(p).exists()]
    if missing:
        raise FileNotFoundError('Missing runtime resources: '+str(missing))
    for name, source in value['links'].items():
        target = ROOT / name
        if target.is_symlink() and target.resolve() == Path(source).resolve():
            continue
        if target.exists() or target.is_symlink():
            raise FileExistsError('Refusing to replace '+str(target))
        if apply:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(Path(source).resolve(), target_is_directory=True)
    return dict(status='configured' if apply else 'checked', links=len(LINKS))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    print(json.dumps(configure(args.config, apply=args.apply)))
