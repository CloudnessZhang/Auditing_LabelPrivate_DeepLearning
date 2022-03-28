from dataclasses import dataclass


###########################################################
# the aim of auditing
###########################################################

@dataclass
class Audit_result:
    audit_function : int = 0
    epsilon_theory: float = .0
    epsilon_lb: float =.0
    epsilon_opt: float =.0
    model_accuary: float =.0
    model_loss:float =.0
    inference_accuary: float =.0
    poisoning_effect: float =.0

###########################################################
# ALIBI args
###########################################################

@dataclass
class ALIBI_LabelPrivacy:
    sigma: float = 1
    max_grad_norm: float = 1e10
    delta: float = 1e-5
    post_process: str = "mapwithprior"
    mechanism: str = "Laplace"
    noise_only_once: bool = True


@dataclass
class ALIBI_Learning:
    lr: float = 0.1
    batch_size: int = 128
    epochs: int = 1
    momentum: float = 0.9
    weight_decay: float = 1e-4
    random_aug: bool = False


@dataclass
class ALIBI_Settings:
    dataset: str = "cifar10"
    arch: str = "resnet"
    privacy: ALIBI_LabelPrivacy = ALIBI_LabelPrivacy()
    learning: ALIBI_Learning = ALIBI_Learning()
    save_dir: str = "../result"
    data_dir: str = "../datasets"
    checkpoint_dir: str = "../checkpoint/"
    seed: int = 11337
    #11337-11347

###########################################################
# LP-MST args
###########################################################
@dataclass
class LPMST_Learning:
    lr: float = 0.1
    batch_size: int = 64
    # epochs: int = 150
    epochs: int = 1
    momentum: float = 0.9
    weight_decay: float = 1e-4

@dataclass
class LPMST_Settings:
    dataset: str = "cifar10"
    epsilon: float = 1.
    # alpha: float = 4.
    learning: LPMST_Learning = LPMST_Learning()
    save_dir: str = "../result"
    data_dir: str = "../datasets"
    checkpoint_dir: str = "../checkpoint/"
    seed: int = 11337


###########################################################
# LP-MST args
###########################################################
import os
from dataclasses import dataclass
from typing import Optional


@dataclass(eq=True, frozen=True)
class PateNoiseConfig:
    selection_noise: float
    result_noise: float
    threshold: int


@dataclass(eq=True, frozen=True)
class CanaryDatasetConfig:
    N: int
    seed: int = 11337


@dataclass(eq=True, frozen=True)
class OptimizerConfig:
    method: str = "SGD"
    method: str = "SGD"
    lr: float = 0.03
    momentum: float = 0.9
    weight_decay: float = 5e-4
    nesterov: bool = True


@dataclass(eq=True, frozen=True)
class LearningConfig:
    optim: OptimizerConfig = OptimizerConfig()
    batch_size: int = 64
    epochs: int = 40


@dataclass(eq=True, frozen=True)
class FixmatchModelConfig:
    width: int = 4
    depth: int = 28
    cardinality: int = 4


@dataclass(eq=True, frozen=True)
class FixmatchConfig:
    model: FixmatchModelConfig = FixmatchModelConfig()
    mu: int = 7
    warmup: float = 0.0
    use_ema: bool = True
    ema_decay: float = 0.999
    amp: bool = False
    opt_level: str = "O1"
    T: float = 1.0
    threshold: float = 0.95
    lambda_u: float = 1.0

@dataclass(eq=True, frozen=True)
class PateStudentConfig:
    noise: PateNoiseConfig
    fixmatch: FixmatchConfig
    n_samples: int = 1000
    learning: LearningConfig = LearningConfig()

    def filename(self):
        noise = self.noise
        setting_str = (
            f"student_noise_{noise.result_noise:.0f}_{noise.selection_noise:.0f}_{noise.threshold}_"
            f"samples_{self.n_samples}_"
            f"epochs_{self.learning.epochs}"
        )

        setting_str += f"_{hash(self):x}"

        return setting_str


@dataclass(eq=True, frozen=True)
class PateTeacherConfig:
    learning: LearningConfig = LearningConfig()
    fixmatch: FixmatchConfig = FixmatchConfig()

@dataclass(eq=True, frozen=True)
class PateCommonConfig:
    n_teachers: int = 800
    dataset: str = "cifar10"
    dataset_root: str = "tmp/cifar10"
    seed: int = 1337
    student_dataset_max_size: int = 10000
    model_dir: str = "tmp/pate"
    canary_dataset: Optional[CanaryDatasetConfig] = None
    tensorboard_log_dir: Optional[str] = None

    # privacy: ALIBI_LabelPrivacy = ALIBI_LabelPrivacy()
    # learning: ALIBI_Learning = ALIBI_Learning()

    @property
    def dataset_dir(self):
        return os.path.join(self.dataset_root, self.dataset)


