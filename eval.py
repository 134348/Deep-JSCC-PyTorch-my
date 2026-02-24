import torch
from utils import get_psnr
import os
from model import DeepJSCC
from train import evaluate_epoch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from dataset import Vanilla
import yaml
from tensorboardX import SummaryWriter
import glob
from concurrent.futures import ProcessPoolExecutor

def eval_snr(model, test_loader, writer, param, times=10):
    snr_list = [0,1,4,7,10,13,19]
    for snr in snr_list:
        model.change_channel(param['channel'], snr)
        test_loss = 0
        for i in range(times):
            test_loss += evaluate_epoch(model, param, test_loader)

        test_loss /= times
        psnr = get_psnr(image=None, gt=None, mse=test_loss)
        writer.add_scalar('psnr', psnr, snr)
        


def process_config(config_path, output_dir, dataset_name, times):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.UnsafeLoader)
        assert dataset_name == config['dataset_name']
        params = config['params']
        c = config['inner_channel']

    if dataset_name == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor(), ])
        test_dataset = datasets.CIFAR10(root='../dataset/', train=False,
                                        download=True, transform=transform)
        test_loader = DataLoader(
            test_dataset, shuffle=False,
            batch_size=params['batch_size'], num_workers=0
        )

    elif dataset_name == 'imagenet':
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((128, 128))])  # the size of paper is 128
        
        test_dataset = Vanilla(root='../dataset/ImageNet/val', transform=transform)
        test_loader = DataLoader(test_dataset, shuffle=True,
                                 batch_size=params['batch_size'], num_workers=params['num_workers'])
    else:
        raise Exception('Unknown dataset')

    name = os.path.splitext(os.path.basename(config_path))[0]
    writer = SummaryWriter(os.path.join(output_dir, 'eval', name))
    model = DeepJSCC(c=c)
    model = model.to(params['device'])
    pkl_list = glob.glob(os.path.join(output_dir, 'checkpoints', name, '*.pkl'))
    model.load_state_dict(torch.load(pkl_list[-1]))
    eval_snr(model, test_loader, writer, params, times)
    writer.close()

def main():
    # 跑一次先出曲线；想更稳再改回 10
    times = 10
    dataset_name = 'cifar10'
    output_dir = './out'

    # 只评测你这次训练出来的 config（避免 Windows 多进程 spawn）
    config_path = os.path.join(
        output_dir, 'configs',
        'CIFAR10_8_10.0_0.17_AWGN_15h03m19s_on_Feb_24_2026.yaml'
    )
    process_config(config_path, output_dir, dataset_name, times)


if __name__ == '__main__':
    main()
