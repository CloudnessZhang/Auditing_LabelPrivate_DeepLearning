# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your root.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import argparse
import os.path

import numpy as np
import torch
from torch.utils.data import Subset, ConcatDataset

import utils
from make_dataset import DataFactory
from args.args_Setting import ALIBI_Settings, Audit_result
from network.model import SUPPORTED_MODEL
from network.alibi_model import ALIBI
from binary_classifier.inference import shadow_model, attack_model, base_MI
from lower_bounding import lowerbound
from sklearn.ensemble import RandomForestClassifier


def main(args):
    assert args.dataset.lower() in DataFactory.SUPPORTED_DATASET, "Unsupported dataset"
    assert args.net.lower() in SUPPORTED_MODEL, "Unsupported model"

    sess = utils.get_sess(args)
    print(sess)

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
    num_classes = DataFactory.K_TABLE[args.dataset.lower()]
    data_factory = DataFactory.Data_Loader(which=args.dataset.lower(), data_root=setting.data_dir)
    train_set, test_set = data_factory.get_train_set(), data_factory.get_test_set()
    if args.audit_function == 0:  # 基于loss error的membership inference 的审计办法
        D_0 = train_set
        D_1 = DataFactory.get_D1(data_factory.get_train_set(), num_classes)
    elif args.audit_function == 1:  # 基于Shadow model的membership inference 的审计办法
        D_0 = train_set
        D_1 = DataFactory.get_D1(data_factory.get_train_set(), num_classes)
        shadow_train_set = D_0
        shadow_test_set = D_1
    elif args.audit_function == 2:
        D_0 = DataFactory.Canaries_Dataset(train_set, num_classes, args.trails, setting.seed) # trials 个标签翻转
        train_set = D_0.getDataset()
        D_1 = data_factory.get_train_set()
    elif args.audit_function == 3:
        D_0 = train_set
        D_1 = DataFactory.get_muti_D1(data_factory.get_train_set(), num_classes)
        shadow_test_set = ConcatDataset(D for D in D_1)
        shadow_train_set = D_0
    elif args.audit_function ==4:
        D_1 = train_set
        if args.net == 'alibi':
            alibi_tmp = ALIBI(trainset=D_1, testset=test_set, num_classes=num_classes, setting=setting)
            model_tmp = alibi_tmp.train_model()
            # model_tmp = alibi_tmp.getModel() #暂时
            D_0 = DataFactory.Poised_Dataset(D=D_1, model=model_tmp, pois_num=1000, pois_func=1)

    ###########################################################
    # 在D0数据集上训练模型，得到模型和理论上的隐私损失
    ###########################################################
    # if args.net == 'alibi':
    #     alibi = ALIBI(train_set, test_set, num_classes, setting)
    #     model = alibi.train_model()
    #     audit_result.epsilon_theory = alibi.get_eps()
    #     audit_result.model_accuary = alibi.acc
    #     audit_result.model_loss = alibi.loss
    #
    # save_name = utils.save_name(args.dataset.lower(), args.net, setting.learning.epochs, audit_result.epsilon_theory,
    #                             args.audit_function)
    # model_path = os.path.join(setting.save_dir, 'model', save_name) + '.pth'
    # # 保存模型
    # torch.save(model, model_path)
    # print("Target模型训练完毕，已保存~")

    # 读取模型， 可以注释上述训练模型过程，直接使用之前训练完成的模型, 读取的epsilon_theory名称自行根据文件名设置
    save_name = utils.save_name(args.dataset.lower(), args.net, setting.learning.epochs, eps_theory=1.0,
                                auditing_function=args.audit_function)
    model_path = os.path.join(setting.save_dir, 'model', save_name) + '.pth'
    model = torch.load(model_path)
    print("Target模型加载完毕~今天也要加油呀！")

    ###########################################################
    # 生成二分类器 ，得到二分类器T
    ###########################################################
    attaker_path = os.path.join(setting.save_dir, 'attacker', save_name) + '.pickle'

    if args.audit_function == 0:
        T = base_MI.BaseMI(train_set, model)
    elif args.audit_function == 1 or args.audit_function==3:  # 基于Shadow model的审计方法
        learner = ALIBI(None, None, num_classes, setting) #resnet18
        shm = shadow_model.ShadowModels(shadow_train_set, shadow_test_set, n_models=5,
                                        target_classes=num_classes, learner=learner,
                                        epochs=setting.learning.epochs,
                                        verbose=0)
        rf_attack = RandomForestClassifier(n_estimators=100)
        T = attack_model.AttackModels(target_classes=10, attack_learner=rf_attack)
        T.fit(shm.results)  # attack model
    elif args.audit_function == 2:
        T = None

    # 保存attacker
    utils.save_Class(T, attaker_path)
    print("Shadow Attacker模型训练完毕，已保存~")

    # # 读取attacker， 可以注释上述训练模型过程，直接使用之前训练完成的模型
    # if args.audit_function == 0:
    #     T = base_MI.BaseMI()
    # elif args.audit_function == 1 or args.audit_function ==3:
    #     T = attack_model.AttackModels()
    # elif args.audit_function == 2:
    #     T = None
    # T = utils.read_Class(T, attaker_path)
    # print("二分类器加载完毕~今天也要加油呀！~")

    # ###########################################################
    # # 计算ε的经验下界
    # ###########################################################
    audit_result_path = os.path.join(setting.save_dir, 'audit_result', save_name) + '.pickle'

    # Result = lowerbound.LowerBound(D_0, D_1, num_classes,model, T, args.audit_function)
    Result = lowerbound.LowerBound(D_0=D_0, D_1=D_1, num_classes=num_classes, model=model, T=T,
                                   audit_function=args.audit_function)

    audit_result.epsilon_opt = Result.eps_OPT
    audit_result.epsilon_lb = Result.eps_LB
    audit_result.inference_accuary = Result.inference_accuary
    audit_result.poisoning_effect = Result.poisoning_effect

    # 输出审计结果
    print(
        save_name,
        f"Model Loss: {audit_result.model_loss} ",
        f"Model Acc: {audit_result.model_accuary} ",
        f"Pois Eff: {audit_result.poisoning_effect} ",
        f"Infer Acc: {audit_result.inference_accuary}\n",
        f"eps Theory: {audit_result.epsilon_theory} ",
        f"eps OPT: {audit_result.epsilon_opt} ",
        f"eps LB: {audit_result.epsilon_lb}"
    )

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
    parser.add_argument('--eps', default=0, type=float, help='privacy parameter epsilon')
    parser.add_argument('--delta', default=1e-5, type=float, help='probability of failure')
    parser.add_argument('--sigma', default=2 * (2.0 ** 0.5) / 1, type=float,
                        help='Guassion or Laplace perturbation coefficient')
    parser.add_argument('--trails', default=10000, type=float, help='The number of sample labels changed is trails')
    parser.add_argument('--audit_function', default=1, type=bool, help='the function of auditing:'
                                                                       '0：based simple inference attack,'
                                                                       '1：based shadow model inference attack'
                                                                       '2: based memorization attack,'
                                                                       '3: based shadow model inference attack with |C-1|'
                                                                       '4：based poisoning attacked,'
                                                                       '5: combined 1 and 2')

    args = parser.parse_args()

    main(args)
