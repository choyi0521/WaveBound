from data_provider.data_loader import DatasetETTHour, DatasetETTMinute, DatasetCustom
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': DatasetETTHour,
    'ETTh2': DatasetETTHour,
    'ETTm1': DatasetETTMinute,
    'ETTm2': DatasetETTMinute,
    'custom': DatasetCustom,
}


def data_provider(args, phase, logger):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        phase=phase,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=args.freq
    )
    
    if phase == 'train':
        data_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    elif phase in ['valid', 'test']:
        data_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    else:
        raise NotImplementedError
    
    logger.info(f'{phase} {len(data_set)}')
    return data_set, data_loader
