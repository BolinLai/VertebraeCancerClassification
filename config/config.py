# coding:utf-8
import warnings


class Config(object):
    env = 'Vertebrae'

    data_root = '/DB/rhome/bllai/Data/VertebraeData'
    train_paths = '/DB/rhome/bllai/PyTorchProjects/Vertebrae/dataset/sup_train_path.csv'
    test_paths = '/DB/rhome/bllai/PyTorchProjects/Vertebrae/dataset/sup_test_path.csv'

    # Used in LSTM-CRF =============================================================
    # data_root = '/DATA5_DB8/data/bllai/Data'
    # train_paths = '/DB/rhome/bllai/PyTorchProjects/Vertebrae/dataset/feature_train_path.csv'
    # test_paths = '/DB/rhome/bllai/PyTorchProjects/Vertebrae/dataset/feature_test_path.csv'
    # ==============================================================================

    data_balance = True
    num_classes = 3

    save_model_name = None
    load_model_path = None

    batch_size = 16
    num_workers = 4
    print_freq = 50

    max_epoch = 40
    lr = 0.0001
    lr_pre = 0.1 * lr
    lr_decay = 0.5
    weight_decay = 1e-5

    use_gpu = True
    # parallel = False
    # num_of_gpu = 2

    results_file = 'results.csv'
    misclassified_file = 'misclassified.csv'

    def parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn('Warning: config has no attribute {}'.format(k))
            setattr(self, k, v)

        print('Use config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))


config = Config()
