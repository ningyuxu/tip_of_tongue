from dataclasses import asdict, dataclass


@dataclass
class Enumberation:
    def choices(self):
        return list(asdict(self).values())


@dataclass
class Split(Enumberation):
    TRAIN: str = "train"
    DEV: str = "dev"
    TEST: str = "test"
