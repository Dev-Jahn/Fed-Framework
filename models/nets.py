from models.cnns import PerceptronModel, FcNet, SimpleCNN, SimpleCNNMNIST, ModerateCNNMNIST, ModerateCNN
from models.resnetcifar import ResNet50_cifar10
from models.vggmodel import vgg11, vgg16
from models.wideresnet import WideResNet


def init_nets(dropout_p, n_parties, args):
    nets = {net_i: None for net_i in range(n_parties)}

    for net_i in range(n_parties):
        if args.model == 'WRN':
            if args.dataset == 'cifar10':
                net = WideResNet(depth=28, num_classes=10, widen_factor=10, dropRate=0.3)
            elif args.dataset == 'cifar100':
                net = WideResNet(depth=28, num_classes=100, widen_factor=10, dropRate=0.3)
            else:
                raise NotImplementedError('Unimplemented')
        elif args.dataset == "generated":
            net = PerceptronModel()
        elif args.model == "mlp":
            if args.dataset == 'covtype':
                input_size = 54
                output_size = 2
                hidden_sizes = [32, 16, 8]
            elif args.dataset == 'a9a':
                input_size = 123
                output_size = 2
                hidden_sizes = [32, 16, 8]
            elif args.dataset == 'rcv1':
                input_size = 47236
                output_size = 2
                hidden_sizes = [32, 16, 8]
            elif args.dataset == 'SUSY':
                input_size = 18
                output_size = 2
                hidden_sizes = [16, 8]
            net = FcNet(input_size, hidden_sizes, output_size, dropout_p)
        elif args.model == "vgg":
            net = vgg11()
        elif args.model == "simple-cnn":
            if args.dataset in ("cifar10", "cinic10", "svhn"):
                net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)
            elif args.dataset in ("mnist", 'femnist', 'fmnist'):
                net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
            elif args.dataset == 'celeba':
                net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=2)
        elif args.model == "vgg-9":
            if args.dataset in ("mnist", 'femnist'):
                net = ModerateCNNMNIST()
            elif args.dataset in ("cifar10", "cinic10", "svhn"):
                # print("in moderate cnn")
                net = ModerateCNN()
            elif args.dataset == 'celeba':
                net = ModerateCNN(output_dim=2)
        elif args.model == "resnet":
            net = ResNet50_cifar10()
        elif args.model == "vgg16":
            net = vgg16()
        else:
            print("not supported yet")
            exit(1)
        nets[net_i] = net

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return nets, model_meta_data, layer_type