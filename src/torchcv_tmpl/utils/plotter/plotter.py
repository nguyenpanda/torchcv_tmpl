from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..log import Logger


class Plotter:

    def __init__(self,
                 history_metric: dict,
                 save_path: str | Path,
                 logger: Logger | None = None,
                 ):
        self.history: dict = history_metric
        self.save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        self.log = logger

    def epoch_series(self,
                     x: str,
                     y: list[tuple[str, str]],
                     img_name: str,
                     ):
        y_names, series = [], []
        try:
            df = pd.DataFrame({'epoch': self.history[x]})

            for k, m in y:
                name = f'{k}_{m}'
                series.append(pd.Series([each_dict[m] for each_dict in self.history[k]]))
                y_names.append(name)

            df = pd.concat([df, *series], axis=1)
        except KeyError as e:
            if self.log:
                self.log(f'{e}: Plotting will be ignore due to wrong key', level='warning')
            return

        df.set_index('epoch', inplace=True)
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=df, marker='o')

        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.xticks(df.index.to_list())
        plt.title(f'Over Epochs')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            self.save_path / f'{img_name}.png',
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()
