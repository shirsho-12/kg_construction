import logging
from config import LOGGING_LEVEL

try:
    import wandb
except ImportError:
    wandb = None


class _BaseLogger(logging.Logger):
    """Simple logger wrapper providing a consistent interface."""

    def log(self, data: dict, level=LOGGING_LEVEL):
        raise NotImplementedError

    def finish(self):
        pass


class _StdLogger(_BaseLogger):
    def __init__(self, name: str, level=LOGGING_LEVEL):
        super().__init__(name)
        self.setLevel(level)
        if not self.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            self.addHandler(ch)


class _WandbLogger(_BaseLogger):
    def __init__(self, project_name: str, run_name: str):
        if wandb is None:
            raise RuntimeError("wandb is not installed")
        self.project_name = project_name
        self.run_name = run_name or "run"
        wandb.init(project=self.project_name, name=self.run_name)

    def log(self, data: dict, level=LOGGING_LEVEL):
        wandb.log(data)

    def finish(self):
        wandb.finish()


class LoggerFactory:
    @staticmethod
    def create(
        project_name: str = "default_project",
        use_wandb: bool = False,
        run_name: str = "",
        level=LOGGING_LEVEL,
    ) -> _BaseLogger:
        """
        Factory that returns a logger implementing log(data: dict) and finish().
        If use_wandb is True and wandb is available, returns a wandb-backed logger.
        Otherwise returns a stdlib logger wrapper.
        """
        if use_wandb:
            if wandb is None:
                # fallback to standard logger if wandb not installed
                logging.getLogger(__name__).warning(
                    "wandb requested but not installed; falling back to std logger"
                )
                use_wandb = False

        if use_wandb:
            return _WandbLogger(project_name=project_name, run_name=run_name)
        else:
            name = f"{project_name}.{run_name or 'run'}"
            return _StdLogger(name=name, level=level)


# convenience alias
create_logger = LoggerFactory.create
logger = create_logger()  # default logger instance
