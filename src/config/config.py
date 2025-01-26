from src.utils import load_yaml_file


class AppConfig:
    def __init__(self, config_path: str = "./config/config.yaml") -> None:
        config_file = load_yaml_file(config_path)
        self.DEVICE: str = config_file["DEVICE"]
        self.INPUT_WIDTH: int = config_file["INPUT_WIDTH"]
        self.INPUT_HEIGHT: int = config_file["INPUT_HEIGHT"]
        self.MODEL_PATH: str = config_file["MODEL_PATH"]
        self.MOCK_DB_PATH: str = config_file["MOCK_DB_PATH"]