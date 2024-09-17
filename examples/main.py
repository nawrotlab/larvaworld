from argparse import ArgumentParser
from runpy import run_path
from pathlib import Path

def get_python_files(parent_folder: str):
    examples = {}
    for path in parent_folder.glob('**/*.py'):
        if 'main' in path.stem:
            continue
        name = path.stem.replace('_', '-')
        examples[name] = path
    return examples


def main():
    parser = ArgumentParser()

    examples = get_python_files(Path(__file__).parent)
    parser.add_argument('name', choices=list(examples.keys()))

    args = parser.parse_args()
    script = examples[args.name]
    run_path(script)

if __name__ == '__main__':
    main()