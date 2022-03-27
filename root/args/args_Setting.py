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