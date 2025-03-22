import logging
from datetime import datetime
from pathlib import Path

from typing_extensions import Literal, get_args

from ..typing import PATH_LIKE

LITERAL_LOG_LEVEL = Literal['debug', 'info', 'warning', 'error', 'critical']
WRITE_MODE = Literal['w', 'a']


class Logger:

    def __init__(self,
                 logger_name: str,
                 path: PATH_LIKE,
                 write_mode: WRITE_MODE = 'a',
                 ):
        self.logger: logging.Logger = logging.getLogger(logger_name)

        self.curr_level = logging.INFO
        self.setLevel(self.curr_level)

        self.path = Path(path)
        if not self.path.match('**/*.*'):
            self.path = self.path / f'{logger_name}_{datetime.today().now().strftime("%Y-%B-%d_%H:%M:%S")}.log'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handlers = self.build_handler(self.path, write_mode)
        self.fmt = self.build_formatter()
        self._add_handlers()

    @classmethod
    def build_handler(cls, file_path: PATH_LIKE, write_mode: WRITE_MODE = 'a') -> list[logging.Handler]:
        return [
            logging.StreamHandler(),
            logging.FileHandler(
                filename=file_path,
                mode=write_mode,
            ),
        ]

    @classmethod
    def build_formatter(cls) -> logging.Formatter:
        return logging.Formatter(
            fmt='%(asctime)s | %(name)s::[%(levelname)8s] | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )

    def log_out(self, message: str, level: LITERAL_LOG_LEVEL = 'info'):
        self(message, level)

    def setLevel(self, level: int | str) -> 'Logger':
        self.logger.setLevel(level)
        return self

    def __call__(self, message: str, level: LITERAL_LOG_LEVEL = 'info'):
        if level == 'info':
            self.logger.info(str(message))
        elif level == 'debug':
            self.logger.debug(message)
        elif level == 'warning':
            self.logger.warning(message)
        elif level == 'error':
            self.logger.error(message)
        elif level == 'critical':
            self.logger.critical(message)
        else:
            raise ValueError(f'level={level} not in {get_args(LITERAL_LOG_LEVEL)}')

    def _add_handlers(self):
        for handler in self.handlers:
            handler.setFormatter(self.fmt)
            self.logger.addHandler(handler)

    def __enter__(self):
        for handler in self.handlers:
            self.logger.removeHandler(handler)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._add_handlers()
        self.setLevel(self.curr_level)


if __name__ == '__main__':
    ROOT_PATH = Path.cwd().parents[1]
    FILE = ROOT_PATH / 'temp'
    log = Logger(__name__, FILE, 'w')
    print(log.handlers[1])
