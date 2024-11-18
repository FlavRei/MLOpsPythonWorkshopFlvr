import json
from abc import abstractmethod, ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List
import random
import numpy as np


class IDataRandom(ABC):
    @abstractmethod
    def shuffle(self, x: list) -> None:
        pass

    @abstractmethod
    def seed(self, seed: int) -> None:
        pass


class IDataManager(ABC):
    @abstractmethod
    def load_json(self, json_path: Path) -> Any:
        pass

    @abstractmethod
    def list_files(self, directory_path: Path, suffix: str) -> List[Path]:
        pass

    @abstractmethod
    def copy_file(self, file_path: Path, to_file_path: Path) -> None:
        pass

    @abstractmethod
    def create_directory(self, directory_path: Path) -> None:
        pass


class DataRandom(IDataRandom):
    def shuffle(self, x: list) -> None:
        random.shuffle(x)

    def seed(self, seed: int) -> None:
        random.seed(seed)


class DataManager(IDataManager):
    def load_json(self, json_path: Path) -> Any:
        with open(json_path) as json_file:
            return json.load(json_file)

    def list_files(self, directory_path: Path, suffix: str) -> List[Path]:
        return [p for p in directory_path.iterdir() if p.is_file() and p.suffix == suffix]

    def copy_file(self, source_path: Path, destination_path: Path) -> None:
        destination_path.write_bytes(source_path.read_bytes())

    def create_directory(self, directory_path: Path) -> None:
        directory_path.mkdir(parents=True, exist_ok=True)


@dataclass
class LabelSplitDataResult:
    number_file_train_by_label: int
    number_file_test_by_label: int
    number_file_evaluate_by_label: int
    number_labeled_data: int


@dataclass
class LabelSplitDataInput:
    input_labels_path: Path
    input_images_directory: Path
    output_images_directory: Path
    number_images_per_label: int
    ratio_train: float = 0.4
    ratio_test: float = 0.4


class DataSplit:
    def __init__(self, data_random: IDataRandom = DataRandom(), data_manager: IDataManager = DataManager()):
        self.data_random = data_random
        self.data_manager = data_manager

    def label_split_data(self, input: LabelSplitDataInput) -> LabelSplitDataResult:
        self.data_random.seed(11)

        if input.ratio_train + input.ratio_test > 1:
            raise ValueError("Sum of train and test ratios must not exceed 1")

        # Chargement des annotations
        annotations = self.data_manager.load_json(input.input_labels_path)["annotations"]

        # Mélange des annotations
        self.data_random.shuffle(annotations)

        # Récupération des fichiers par label
        split_paths = {label: [] for label in ["oui", "non", "autre"]}
        for annotation in annotations:
            filename = annotation["fileName"]
            label = annotation["annotation"]["label"]
            if len(split_paths[label]) < input.number_images_per_label:
                split_paths[label].append(filename)

        # Vérification du nombre minimum d'images par label
        for label, files in split_paths.items():
            if len(files) < input.number_images_per_label:
                raise ValueError(f"Not enough files for label '{label}'")

        # Répartition des fichiers
        num_train = int(input.number_images_per_label * input.ratio_train)
        num_test = int(input.number_images_per_label * input.ratio_test)

        for label, files in split_paths.items():
            train_files, test_files, eval_files = np.split(
                files, [num_train, num_train + num_test]
            )

            for split_name, file_list in zip(
                ["train", "test", "evaluate"], [train_files, test_files, eval_files]
            ):
                output_dir = input.output_images_directory / split_name / f"{label}s"
                self.data_manager.create_directory(output_dir)

                for filename in file_list:
                    src_path = input.input_images_directory / filename
                    dest_path = output_dir / filename
                    self.data_manager.copy_file(src_path, dest_path)

        # Résultat
        return LabelSplitDataResult(
            number_file_train_by_label=num_train,
            number_file_test_by_label=num_test,
            number_file_evaluate_by_label=input.number_images_per_label - num_train - num_test,
            number_labeled_data=len(annotations),
        )
