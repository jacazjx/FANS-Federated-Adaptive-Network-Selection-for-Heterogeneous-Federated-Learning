import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default="cifar10")
parser.add_argument('--ratio', type=float, default=0.1)
parser.add_argument('--csv_path', type=str, default="resnet_result.csv")
args = parser.parse_args()


if args.task == "cifar10":
    from hypernet.hypernetworks.resnet import super_resnet18, eval_all_subnets

    #############################################################
    ## eval cifar-10
    #############################################################
    import time
    # time.sleep(3*60*30)
    resnet = torch.load(f"logs/cifar10/seed4321/hypernet_{args.ratio}.pth").cuda()
    # resnet = super_resnet18(True, False, [1, 0.75, 0.5, 0.25], 10).cuda()
    # resnet.load_state_dict(para.state_dict())
    eval_all_subnets(resnet, "../data", args.csv_path)

elif args.task == "cifar100":
    from hypernet.hypernetworks.densenet import super_densenet121, eval_all_subnets
    ############################################################
    # eval cifar-100
    ############################################################

    para = torch.load(f"logs/cifar100/seed4321/hypernet_{args.ratio}.pth").cuda()
    resnet = super_densenet121(num_classes=100).cuda()
    resnet.load_state_dict(para.state_dict())
    eval_all_subnets(resnet, "../data", args.csv_path)



