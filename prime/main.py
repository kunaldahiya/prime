from models.network import SiameseNetworkIS, XCNetworkIS


def construct_network(args):
    if args.stage == 'siamese':
        return SiameseNetworkIS.from_config(
            config=args.arch, 
            device="cuda", 
            args=args
        )    
    elif args.stage == 'xc':
        return XCNetworkIS.from_config(
            config=args.arch, 
            device="cuda", 
            args=args
        )    
    else:
        raise NotImplementedError("")


def main(args):
    net = construct_network(args)
    print(net)


if __name__ == "__main__":
    pass