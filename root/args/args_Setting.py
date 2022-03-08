from dataclasses import dataclass


###########################################################
# the aim of auditing
###########################################################

@dataclass
class Audit_result:
    audit_function : int = 0
    epsilon_theory: float = .0
    epsilon_LB: float =.0
    epsilon_OPT: float =.0
    model_Accuary: float =.0
    model_Loss:float =.0
    inference_Accuary: float =.0
    Poisoning_Effect: float =.0

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
    pois_num: int = 1000
    arch: str = "resnet"
    privacy: ALIBI_LabelPrivacy = ALIBI_LabelPrivacy()
    learning: ALIBI_Learning = ALIBI_Learning()
    save_dir: str = "../result"
    data_dir: str = "../datasets"
    checkpoint_dir: str = "../checkpoint/"
    seed: int = 11337