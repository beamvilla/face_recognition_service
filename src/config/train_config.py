from utils import load_yaml_file


class TrainConfig:
    def __init__(self, config_path: str = "./config/train.yaml") -> None:
        config_file = load_yaml_file(config_path)
        self.EPOCHS: int = config_file["EPOCHS"]
        self.TRAIN_DIR: str = config_file["TRAIN_DIR"]
        self.TEST_DIR: str = config_file["TEST_DIR"]
        self.MODEL_DIR: str = config_file["MODEL_DIR"]
        self.LOSS_ALPHA: float = config_file["LOSS_ALPHA"]
        self.LEARNING_RATE: float = float(config_file["LEARNING_RATE"])
        self.DEVICE: str = config_file["DEVICE"]
        self.INPUT_WIDTH: int = config_file["INPUT_WIDTH"]
        self.INPUT_HEIGHT: int = config_file["INPUT_HEIGHT"]