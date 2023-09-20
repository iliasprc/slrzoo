import argparse


def weakly_supervised_training_arguments():
    parser = argparse.ArgumentParser(description='SLR weakly supervised training')
    parser.add_argument('--modality', type=str, default='full', metavar='rc',
                        help='hands or full image')
    parser.add_argument('--dataset', type=str, default='phoenix2014', metavar='rc',
                        help='slr dataset')

    parser.add_argument('--model', type=str, default='subunet', help='subunet or cui or i3d or sfnet')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')

    parser.add_argument('--epochs', type=int, default=400, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')

    parser.add_argument('--cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--supervised', action='store_true', default=False
                        )
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1234)')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='image normalization mean std')
    parser.add_argument('--run_full', action='store_true', default=True
                        )
    parser.add_argument('--padding', action='store_true', default=False,
                        help='video padding')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='For Saving the current Model')

    parser.add_argument('--seq-length', type=int, default=250, metavar='num', help='frame sequence length')
    parser.add_argument('--hidden_size', type=int, default=1024, metavar='num', help='lstm units')
    parser.add_argument('--optim', type=str, default='adam', metavar='optim number', help='optimizer sgd or adam')
    parser.add_argument('--n_layers', type=int, default=2, metavar='num', help='rnn layers')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='num', help='hidden size')
    parser.add_argument('--bidirectional', action='store_true', default=True, help='bidirectional for rnn')
    parser.add_argument('--cnn', type=str, default='alexnet', help='cnn backbone')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--ctc', type=str, default='normal',
                        help='normal for vanilla-CTC or focal or ent_ctc or custom or weighted or aggregation or stim_ctc')

    args = parser.parse_args()
    return args


def fully_supervised_training_arguments():
    parser = argparse.ArgumentParser(description='SLR fully supervised')
    parser.add_argument('--modality', type=str, default='full', metavar='rc',
                        help='hands or full image')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='image normalization mean std')
    parser.add_argument('--dataset', type=str, default='phoenix_iso_cui', metavar='rc',
                        help='slr dataset phoenix_iso, phoenix_iso_I5, ms_asl , signum_isolated , csl')

    parser.add_argument('--model', type=str, default='cui', help='subunet or cui or i3d ')
    parser.add_argument('--padding', action='store_true', default=True,
                        help='image normalization mean std')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 1)')

    parser.add_argument('--epochs', type=int, default=400, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')

    parser.add_argument('--cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--run_full', action='store_true', default=False
                        )
    parser.add_argument('--supervised', action='store_true', default=True
                        )
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1234)')
    parser.add_argument('--log-interval', type=int, default=2000, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='For Saving the current Model')

    parser.add_argument('--seq-length', type=int, default=16, metavar='num',
                        help='frame sequence length 16 for phoenix , 64 for MS_ASL')
    parser.add_argument('--hidden_size', type=int, default=512, metavar='num', help='lstm units')
    parser.add_argument('--optim', type=str, default='adam', metavar='optim number', help='optimizer sgd or adam')
    parser.add_argument('--n_layers', type=int, default=2, metavar='num', help='rnn layers')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='num', help='hidden size')
    parser.add_argument('--bidirectional', action='store_true', default=True, help='bidirectional for rnn')
    parser.add_argument('--cnn', type=str, default='googlenet', help='cnn backbone')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--rnn', type=str, default='tcn', help='cnn backbone')
    args = parser.parse_args()
    return args


def main_2stream_training_arguments():
    parser = argparse.ArgumentParser(description='SLR')
    parser.add_argument('--modality', type=str, default='full_image', metavar='rc',
                        help='hands or full image')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
                        help='learning rate (default: 0.00001)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dataloader', type=str, default='rand_crop', metavar='rc',
                        help='data augmentation')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--seq-length', type=int, default=250, metavar='num', help='squence length')
    parser.add_argument('--hidden_size', type=int, default=1024, metavar='num', help='lstm units')
    parser.add_argument('--optim', type=str, default='adam', metavar='optim number', help='optimizer sgd or adam')
    parser.add_argument('--n_layers', type=int, default=2, metavar='num', help='hidden size')
    parser.add_argument('--dropout', type=float, default=0.7, metavar='num', help='hidden size')
    parser.add_argument('--bidirectional', action='store_true', default=True, help='hidden size')
    parser.add_argument('--cnn', type=str, default='alexnet')
    parser.add_argument('--gpu', type=str, default='1')
    args = parser.parse_args()
    return args
