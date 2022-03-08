# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your root.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import argparse
import os.path

import numpy as np
import torch
from torch.utils.data import Subset

import utils
from make_dataset.DataFactory import SUPPORTED_DATASET, DataFactory, K_TABLE
from args.args_Setting import ALIBI_Settings, Audit_result
from network.model import SUPPORTED_MODEL
from network.alibi_model import ALIBI
from binary_classifier.inference import shadow_model, attack_model, base_MI
from lower_bounding import lowerbound_mi
from sklearn.ensemble import RandomForestClassifier


def main(args):
    assert args.dataset.lower() in SUPPORTED_DATASET, "Unsupported dataset"
    assert args.net.lower() in SUPPORTED_MODEL, "Unsupported model"

    sess = utils.get_sess(args)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    audit_result = Audit_result
    audit_result.audit_function = args.audit_function
    if args.net.lower() == 'alibi':
        setting = ALIBI_Settings
        setting.dataset = args.dataset.lower()
        setting.privacy.sigma = args.sigma
        setting.privacy.delta = args.delta

    utils.make_deterministic(setting.seed)  # 随机种子设置
    ###########################################################
    # 加载数据集D0，D1
    ###########################################################
    num_classes = K_TABLE[args.dataset.lower()]
    data_factory = DataFactory(which=args.dataset.lower(), data_root=setting.data_dir)
    train_set, test_set = data_factory.get_train_set(), data_factory.get_test_set()
    if args.audit_function == 0:  # 基于loss mean的membership inference 的审计办法
        D_0 = train_set
        D_1 = test_set
    elif args.audit_function == 1:  # 基于Shadow model的membership inference 的审计办法
        target_train_set = Subset(train_set, range(0, 25000))
        shadow_train_set = Subset(train_set, range(25000, 50000))
        target_test_set = Subset(test_set, range(0, 5000))
        shadow_test_set = Subset(test_set, range(5000, 10000))
        D_0 = target_train_set
        D_1 = target_test_set
    elif args.audit_function == 2:
        D_1 = train_set
        if args.net == 'alibi':
            alibi = ALIBI(sess, D_1, num_classes, setting)
            model = alibi.train_model()
        # D_0 = poisoned(D_1,net)

    ###########################################################
    # 在D0数据集上训练模型，得到模型和理论上的隐私损失
    ###########################################################
    if args.net == 'alibi':
        alibi = ALIBI(D_0, D_1, num_classes, setting)
        model = alibi.train_model()
        audit_result.epsilon_theory = alibi.get_eps()
        audit_result.model_Accuary = alibi.acc
        audit_result.model_Loss = alibi.loss

    save_name = utils.save_name(args.dataset.lower(), args.net, setting.learning.epochs, audit_result.epsilon_theory,
                                args.audit_function)
    model_path = os.path.join(setting.save_dir, 'model', save_name) + '.pth'
    # 保存模型
    torch.save(model, model_path)
    print("Target模型训练完毕，已保存~")

    # # 读取模型， 可以注释上述训练模型过程，直接使用之前训练完成的模型, 读取的epsilon_theory名称自行根据文件名设置
    # save_name = utils.save_name(args.dataset.lower(), args.net, setting.learning.epochs, epsilon_theory=0,
    #                             auditing_function=args.audit_function)
    # model_path = os.path.join(setting.save_dir, 'model', save_name) + '.pth'
    # model = torch.load(model_path)
    # print("Target模型加载完毕~今天也要加油呀！")

    ###########################################################
    # 生成二分类器 ，得到二分类器T
    ###########################################################
    attaker_path = os.path.join(setting.save_dir, 'attacker', save_name) + '.pickle'

    if args.audit_function == 0:
        T = base_MI(D_0, model)
    elif args.audit_function == 1:  # 基于Shadow model的审计方法
        alibi_shadow = ALIBI(sess, None, num_classes, setting)
        shm = shadow_model.ShadowModels(shadow_train_set, shadow_test_set, n_models=5,
                                        target_classes=num_classes, learner=alibi_shadow,
                                        epochs=setting.learning.epochs,
                                        verbose=0)
        rf_attack = RandomForestClassifier(n_estimators=100)
        T = attack_model.AttackModels(target_classes=10, attack_learner=rf_attack)
        T.fit(shm.results)  # attack model

        # 保存attacker
        utils.save_Class(T, attaker_path)
        print("Shadow Attacker模型训练完毕，已保存~")

    # # 读取attacker， 可以注释上述训练模型过程，直接使用之前训练完成的模型
    # if args.audit_function == 0:
    #     T = base_MI()
    # elif args.audit_function == 1:
    #     T = attack_model.AttackModels()
    # T = utils.read_Class(T, attaker_path)
    # print("Shadow Attacker模型加载完毕~今天也要加油呀！~")

    # ###########################################################
    # # 计算ε的经验下界
    # ###########################################################
    audit_result_path = os.path.join(setting.save_dir, 'audit_result', save_name) + '.pickle'

    Result = lowerbound_mi.LowerBound(D_0, D_1, model, T, args.audit_function)
    audit_result.epsilon_OPT = Result.EPS_LB.eps_OPT
    audit_result.epsilon_LB = Result.EPS_LB.eps_LB

    # 保存audit_result
    utils.save_Class(audit_result, audit_result_path)
    print("审计已完成，已保存~")

    # # # 读取audit_result
    # audit_result = Audit_result()
    # audit_result = utils.read_Class(audit_result, audit_result_path)
    # print("audit result加载完毕~今天也要加油呀！~")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Auditing Label Private Deep Learning')  # argparse 命令行参数解析器

    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset name')
    parser.add_argument('--net', default='alibi', type=str, help='label private deep learning to be audited')
    parser.add_argument('--resume', default=False, type=bool, help='resume from checkpoint')
    parser.add_argument('--eps', default=0, type=float, help='privacy parameter epsilon')
    parser.add_argument('--delta', default=1e-5, type=float, help='probability of failure')
    parser.add_argument('--sigma', default=2*(2.0 ** 0.5)/1, type=float, help='Guassion or Laplace perturbation coefficient')
    parser.add_argument('--audit_function', default=0, type=bool, help='the function of auditing:'
                                                                       '0：based simple inference attack,'
                                                                       '1: based shadow model inference attack,'
                                                                       '2：based poisoning attacked,'
                                                                       '3: combined 1 and 2')

    args = parser.parse_args()

    main(args)
    # 默认为'mnist_e20_b265_2st_alpha4_eps8.0_d2_gamma0'T
