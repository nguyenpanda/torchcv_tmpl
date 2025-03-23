from pathlib import Path

from nguyenpanda.swan import green, red

from trainer import MnistCsvTrainer


def unzip(src: str | Path, des: str | Path):
    import zipfile

    print(f'Unzipping "{green(src)}" and save to "{green(des)}"')
    with zipfile.ZipFile(src, 'r') as zip_ref:
        zip_ref.extractall(des)
    print(f'Successfully unzip {src}')


def auto_download_dataset(url: str, path: str | Path):
    import requests

    print(f'Downloading dataset from {green(url)}')
    response = requests.get(url)
    zip_path = Path(path) / 'MNIST_CSV.zip'

    if response.status_code != 200:
        mess = f'Failed to download dataset. Status code: {response.status_code}'
        print(red(mess))
        raise requests.HTTPError(mess)

    with open(zip_path, 'wb') as f:
        for chuck in response.iter_content(chunk_size=1024 * 10):
            if chuck:
                f.write(chuck)
        print(f'Dataset downloaded successfully as {path}')

    unzip(zip_path, path)


if __name__ == '__main__':
    CWD = Path.cwd()
    DEMO = Path(__file__).parents[2]
    MNIST_CSV = DEMO / 'dataset' / 'MNIST_CSV'

    TRAIN_CSV = MNIST_CSV / 'mnist_train.csv'
    TEST_CSV = MNIST_CSV / 'mnist_test.csv'

    if not MNIST_CSV.is_dir():
        MNIST_CSV.mkdir(parents=True)
        auto_download_dataset(
            'https://www.kaggle.com/api/v1/datasets/download/oddrationale/mnist-in-csv',
            MNIST_CSV,
        )

    mnist_trainer = MnistCsvTrainer(
        csv_train_path=TRAIN_CSV,  # : str | Path,
        csv_validate_path=TEST_CSV,  # : str | Path,
        max_epoch=10,  # : int,
        batch_size=32,  # : int,
        learning_rate=0.001,  # : float = 0.001,
        save_folder_path=CWD / 'training_history',  # : Optional[Path | str] = None,
        validate_period=2,  # : Optional[int] = None,
    )

    history, best_validate = mnist_trainer.train()
