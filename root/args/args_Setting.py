from dataclasses import dataclass


###########################################################
# the aim of auditing
###########################################################

@dataclass
class Epsilon_Bound:
    epsilon_theory: float
    epsilon_LB: float
    epsilon_OPT: float
    epsilon_infer: float


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
    epochs: int = 10
    momentum: float = 0.9
    weight_decay: float = 1e-4
    random_aug: bool = False


@dataclass
class ALIBI_Settings:
    dataset: str = "cifar100"
    canary: int = 0
    arch: str = "wide-resnet"
    privacy: ALIBI_LabelPrivacy = ALIBI_LabelPrivacy()
    learning: ALIBI_Learning = ALIBI_Learning()
    gpu: int = -1
    world_size: int = 1
    save_dir: str = "../result"
    data_dir: str = "../datasets"
    checkpoint_dir: str = "../checkpoint/"
    seed: int = 11337
    audit_result = Epsilon_Bound
