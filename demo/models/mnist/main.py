from pathlib import Path

from trainer import MnistCsvTrainer

if __name__ == '__main__':
    CWD = Path.cwd()
    DEMO = Path(__file__).parents[2]
    MNIST_CSV = DEMO / 'dataset' / 'MNIST_CSV'

    TRAIN_CSV = MNIST_CSV / 'mnist_train.csv'
    TEST_CSV = MNIST_CSV / 'mnist_test.csv'

    mnist_trainer = MnistCsvTrainer(
        csv_train_path=TRAIN_CSV,  # : str | Path,
        csv_validate_path=TEST_CSV,  # : str | Path,
        max_epoch=10,  # : int,
        batch_size=32,  # : int,
        learning_rate=0.001,  # : float = 0.001,
        save_folder_path=CWD / 'training_history',  # : Optional[Path | str] = None,
        validate_period=2,  # : Optional[int] = None,
    )

    mnist_trainer.train()
