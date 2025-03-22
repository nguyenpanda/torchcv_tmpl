from pathlib import Path


def append_name(name: str | Path, added) -> str | Path:
    added = str(added)

    if isinstance(name, str):
        if '.' in name:
            name, extension = name.rsplit('.', maxsplit=1)
            added += ('.' + extension)
        return name + added
    elif isinstance(name, Path):
        return Path(name.absolute().parent) / (name.stem + added + name.suffix)

    raise NotImplementedError


def auto_naming(save_folder_path: Path, name: str, sep: str = '-') -> Path:
    save_folder_path = Path(save_folder_path)
    all_idx = tuple(map(
        lambda p: int(p.stem.rsplit(sep, 1)[1]),
        save_folder_path.glob(f'{name}*'),
    ))
    idx = int(max(all_idx) + 1 if all_idx else 0)
    return save_folder_path / (name + str(idx))


if __name__ == '__main__':
    a = 'history'
    b = 'main.py'
    c = Path.cwd() / 'history'
    d = Path.cwd() / 'history' / 'main.py'

    print(a, append_name(a, '_1'))
    print(b, append_name(b, '_1'))
    print(c, append_name(c, '_1'))
    print(d, append_name(d, '_1'))
