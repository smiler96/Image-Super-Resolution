import numpy as np

DIV2K_Statics = {'mean': (0.4488, 0.4371, 0.4040),
                 'std': (1.0, 1.0, 1.0)}

def normalize_(x, type=0):
    # assert type(x) is np.ndarray
    assert len(x.shape) == 3 and x.shape[2] == 3

    x = x.astype(np.float32)
    if type==0:
        pass
    elif type==1:
        x = x / 255.0
    elif type==2:
        x[:, :, 0] = (x[:, :, 0] - (DIV2K_Statics['mean'][0] * 255.0)) / DIV2K_Statics['std'][0]
        x[:, :, 1] = (x[:, :, 1] - (DIV2K_Statics['mean'][1] * 255.0)) / DIV2K_Statics['std'][1]
        x[:, :, 2] = (x[:, :, 2] - (DIV2K_Statics['mean'][2] * 255.0)) / DIV2K_Statics['std'][2]
    else:
        raise NotImplementedError

    return x

def denormalize_(x, type=0):
    # assert type(x) is np.ndarray
    assert len(x.shape) == 3 and x.shape[2] == 3
    x = x.astype(np.float32)
    if type == 0:
        pass
    elif type == 1:
        x = x * 255.0
    elif type == 2:
        x[:, :, 0] = (x[:, :, 0] * DIV2K_Statics['std'][0]) + (DIV2K_Statics['mean'][0] * 255.0)
        x[:, :, 1] = (x[:, :, 1] * DIV2K_Statics['std'][1]) + (DIV2K_Statics['mean'][1] * 255.0)
        x[:, :, 2] = (x[:, :, 2] * DIV2K_Statics['std'][2]) + (DIV2K_Statics['mean'][2] * 255.0)
    else:
        raise NotImplementedError

    return x

def log_status(file, **kwargs):
    status = ''
    for k, v in kwargs.items():
        status += f'{k}: {v}\n'
    with open(file, 'w') as f:
        f.write(status)
    # print(kwargs)

def load_status(file):
    status = {}
    with open(file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            line = line.split(': ')
            key = line[0]
            val = line[1]
            if key == 'epoch':
                val = int(val)
            elif key == 'lr' or key == 'best_val_loss':
                val = float(val)
            status[key] = val

    return status