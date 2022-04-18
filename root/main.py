# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your root.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import argparse
import os.path
import torch

import utils
from make_dataset import DataFactory
from args.args_Setting import ALIBI_Settings, Audit_result, LPMST_Settings
from make_dataset.DataFactory import Data_Loader
from network.model import SUPPORTED_MODEL, ResNet18, AttackModel
from network.alibi_model import ALIBI
from network.lpmst_model import LPMST
from binary_classifier.inference import shadow_model, attack_model, base_MI
from lower_bounding import lowerbound
from sklearn.ensemble import RandomForestClassifier
import time


def main_work(args):
    start = time.time()

    assert args.dataset.lower() in DataFactory.SUPPORTED_DATASET, "Unsupported dataset"
    assert args.net.lower() in SUPPORTED_MODEL, "Unsupported model"

    audit_result = Audit_result
    audit_result.making_datasets_method = args.making_datasets
    audit_result.binary_classifier_method = args.binary_classifier

    num_classes = DataFactory.K_TABLE[args.dataset.lower()]

    if args.net.lower() == 'alibi':
        setting = ALIBI_Settings
        setting.dataset = args.dataset.lower()
        setting.privacy.sigma = 2 * (2.0 ** 0.5) / args.eps
        setting.privacy.delta = args.delta
        setting.learning.epochs = args.epoch
    elif args.net.lower() == 'lp-mst':
        setting = LPMST_Settings
        setting.dataset = args.dataset.lower()
        setting.epsilon = args.eps
        setting.learning.epochs = args.epoch

    # save path
    save_name = utils.save_name(setting.learning.epochs, args)
    datasets_path = os.path.join(setting.save_dir, 'AdjacentDatasets', save_name) + '.pickle'
    model_path = os.path.join(setting.save_dir, 'TrainedModel', save_name) + '.pth'
    classifier_path = os.path.join(setting.save_dir, 'BinaryClassifier', save_name) + '.pickle'
    audit_result_path = os.path.join(setting.save_dir, 'AuditResult', save_name) + '.pickle'
    result_xlsx_path = os.path.join(setting.save_dir, 'Result.xlsx')
    print(save_name)

    utils.make_deterministic(setting.seed)  # 随机种子设置
    ###########################################################
    # 加载数据集D0，D1
    ###########################################################
    data_make = DataFactory.Data_Factory(args=args, setting=setting, num_classes=num_classes, audit_result=audit_result)
    if args.resume > 0:  # resume
        data_make = utils.read_Class(data_make, datasets_path)
        prior_model = torch.load(model_path)
        print("相邻数据集加载完毕~今天也要加油呀！")
    else:
        prior_model = data_make.make_dataset()
        # utils.save_Class(data_make, datasets_path)
        # torch.save(prior_model, model_path)
        # print("相邻数据集已生成，已保存~")
    train_set, test_set, D_0, D_1, shadow_train_set, shadow_test_set, poisoning_set, audit_result = data_make.get_data()
    torch.cuda.empty_cache()

    ###########################################################
    # 在D0数据集上训练模型，得到模型和理论上的隐私损失: 输入train_set 和 test_set
    ###########################################################

    if args.binary_classifier == 1:  # Memorization 使用poisoning dataset训练
        assert args.making_datasets != 0, "Memorization can not using simple datasets making"
        train_set = poisoning_set
        D_0, D_1 = D_1, D_0
        shadow_train_set = poisoning_set
        shadow_test_set = D_1
        if args.net.lower() == 'lp-mst':
            train_set = utils.partition(train_set, 2)
            shadow_train_set = train_set[1]

    if args.resume > 1:  # resume
        model = torch.load(model_path)
        audit_result = utils.read_Class(audit_result, audit_result_path)
        print("Label Private Deep Learning 模型加载完毕~今天也要加油呀！")
    else:
        if prior_model is not None:  # poisoning attack already trained model
            model = prior_model
        else:
            if args.net == 'alibi':
                label_model = ALIBI(train_set, test_set, num_classes, setting, args)
            elif args.net == 'lp-mst':
                label_model = LPMST(train_set, test_set, num_classes, setting, args)
            model = label_model.train_model()
            audit_result.model_accuary = label_model.acc
            audit_result.model_loss = label_model.loss

        # torch.save(model, model_path)
        # utils.save_Class(audit_result, audit_result_path)
        print("Label Private Deep Learning 模型训练完毕，已保存~")
    torch.cuda.empty_cache()

    ###########################################################
    # 生成二分类器 ，得到二分类器T:输入train_set 和 D_1
    ###########################################################

    if args.resume > 2:  # resume
        if args.binary_classifier == 0:
            T = base_MI.BaseMI()
        elif args.binary_classifier == 1:
            T = None
        elif args.binary_classifier == 2:
            T = attack_model.AttackModels()
        T = utils.read_Class(T, classifier_path)
        print("Binary Classifier加载完毕~今天也要加油呀！~")
    else:
        if args.binary_classifier == 0:  # Simple_MI
            if args.net == 'lp-mst':
                T = base_MI.BaseMI(D_train=train_set[1], dataname=args.dataset.lower(), net=model)
            else:
                T = base_MI.BaseMI(D_train=train_set, dataname=args.dataset.lower(), net=model)
        elif args.binary_classifier == 1:  # Memorization attack
            T = None
        elif args.binary_classifier == 2:  # Shadow Model
            # learner = ALIBI(None, None, num_classes, setting)  # resnet18
            learner = ResNet18(num_classes, setting)
            shm = shadow_model.ShadowModels(args.dataset.lower(), shadow_train_set, shadow_test_set, n_models=5,
                                            target_classes=num_classes, learner=learner,
                                            epochs=setting.learning.epochs,
                                            verbose=0)
            rf_attack = RandomForestClassifier(n_estimators=100)
            T = attack_model.AttackModels(target_classes=num_classes, attack_learner=rf_attack)
            T.fit(shm.results)  # attack model
        # utils.save_Class(T, classifier_path)
        print("Binary Classifier训练完毕，已保存~")

    # ###########################################################
    # # 计算ε的经验下界
    # ###########################################################

    if args.resume == 4:
        audit_result = utils.read_Class(audit_result, audit_result_path)
        print("audit result加载完毕~今天也要加油呀！~")
    else:
        Result = lowerbound.LowerBound(D_0=D_0, D_1=D_1, num_classes=num_classes, model=model, T=T, args=args,
                                       epsilon_theory=args.eps)
        audit_result.epsilon_opt = Result.eps_OPT
        audit_result.epsilon_lb = Result.eps_LB
        audit_result.inference_accuary = Result.inference_accuary
        # utils.save_Class(audit_result, audit_result_path)
        end = time.time()
        audit_result.time = float(end - start) / 60
        print("审计已完成，已保存~")

    # 输出审计结果
    print(
        save_name,
        f"Model Loss: {audit_result.model_loss} ",
        f"Model Acc: {audit_result.model_accuary} ",
        f"Infer Acc: {audit_result.inference_accuary}\n",
        f"eps OPT: {audit_result.epsilon_opt} ",
        f"eps LB: {audit_result.epsilon_lb}",
        f"time:{audit_result.time}"
    )

    utils.write_xlsx(result_xlsx_path, args, audit_result)


def set_args(args, making_datasets=None, binary_classifier=None, eps=None, dataset=None, net=None, epoch=None):
    if making_datasets is not None:
        args.making_datasets = making_datasets
    if binary_classifier is not None:
        args.binary_classifier = binary_classifier
    if eps is not None:
        args.eps = eps
    if dataset is not None:
        args.dataset = dataset
    if net is not None:
        args.net = net
    if epoch is not None:
        args.epoch = epoch
    return args


def main(args):
    if args.runAll:
        for dataset in ['mnist', 'cifar10', 'cifar100']:
            args.dataset = dataset
            if dataset == 'mnist':
                for eps in [1, 2, 4, 10]:
                    main_work(set_args(eps=eps, epoch=40))
                    torch.cuda.empty_cache()
            else:
                for eps in [1, 2, 4, 10]:
                    main_work(set_args(eps=eps, epoch=200))
                    torch.cuda.empty_cache()
    else:
        main_work(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Auditing Label Private Deep Learning')  # argparse 命令行参数解析器
    parser.add_argument('--runAll', default=False, type=bool)
    parser.add_argument('--resume', default=0, type=int, help='where to star resume'
                                                              '0: do not need to resume'
                                                              '1: resume datasets'
                                                              '2: resume datasets and model'
                                                              '3: resume datasets, model and binary classifier'
                                                              '4: all resume')
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset name')
    parser.add_argument('--net', default='alibi', type=str, help='label private deep learning to be audited')
    parser.add_argument('--epoch', default=40, type=int, help='the epoch model trains')
    parser.add_argument('--eps', default=4, type=float, help='privacy parameter epsilon')
    parser.add_argument('--delta', default=1e-5, type=float, help='probability of failure')
    parser.add_argument('--trials', default=5000, type=float, help='The number of sample labels changed is trials')

    parser.add_argument('--making_datasets', default=2, type=int, help='the function of making datasets:'
                                                                       '0: simple datasets masking'
                                                                       '1: flipping attack'
                                                                       '2: poisoning attack')
    parser.add_argument('--binary_classifier', default=2, type=int,
                        help='the binary classifier to be combined with poisoned attack:'
                             '0: simple inference attack,'
                             '1: memorization attack,'
                             '2: shadow model inference attack')
    parser.add_argument('--classed_random', default=True, type=bool, help='Whether to poison a specific target')

    args = parser.parse_args()

    main(args)
