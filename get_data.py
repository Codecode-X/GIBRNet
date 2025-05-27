import mydataset
import torch
import torch.utils.data
import random
import collate_fn

def get_data(args):
    f = open("./data/change_sample.txt","r")
    atomic_dict = dict(eval(f.read()))

    train_dict={'single': list(), 'mutual': list(), 'avert': list(), 'refer': list(), 'follow': list(), 'share': list()}
    val_dict = {'single': list(), 'mutual': list(), 'avert': list(), 'refer': list(), 'follow': list(), 'share': list()}
    test_dict = {'single': list(), 'mutual': list(), 'avert': list(), 'refer': list(), 'follow': list(), 'share': list()}

    val_seq=[]
    test_seq=[]

    for mode in atomic_dict.keys(): 
        L=len(atomic_dict[mode])
        print('{}_len:'.format(mode),L)

        train_dict[mode].extend(atomic_dict[mode][:L*2//3])
        val_dict[mode].extend(atomic_dict[mode][:])
        test_dict[mode].extend(atomic_dict[mode][L*2//3:])

        random.shuffle(train_dict[mode]) 

    for mode in atomic_dict.keys():
        test_seq.extend(test_dict[mode])


    train_set=mydataset.mydataset(train_dict,is_train=True)
    val_set=mydataset.mydataset(val_dict, is_train=False)
    test_set = mydataset.mydataset(test_dict, is_train=False)

    train_loader=torch.utils.data.DataLoader(train_set,collate_fn=collate_fn.collate_fn, batch_size=args.batch_size, shuffle=False)
    val_loader=torch.utils.data.DataLoader(val_set,collate_fn=collate_fn.collate_fn, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, collate_fn=collate_fn.collate_fn, batch_size=args.batch_size,  shuffle=False)


    print('Datset sizes: {} training, {} validation, {} testing.'.format(len(train_loader),len(val_loader),len(test_loader)))

    return train_set, val_set, test_set, train_loader, val_loader, test_loader