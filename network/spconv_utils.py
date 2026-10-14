import spconv.pytorch as spconv

def conv3x3(in_planes, out_planes, stride=1, indice_key=None):
    indice_key = f"{indice_key}_3x3" if indice_key is not None else None
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                             padding=1, bias=False, indice_key=indice_key)


def conv1x3(in_planes, out_planes, stride=1, indice_key=None):
    indice_key = f"{indice_key}_1x3" if indice_key is not None else None
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=stride,
                             padding=(0, 1, 1), bias=False, indice_key=indice_key)

def conv3x1(in_planes, out_planes, stride=1, indice_key=None):
    indice_key = f"{indice_key}_3x1" if indice_key is not None else None
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(3, 1, 3), stride=stride,
                             padding=(1, 0, 1), bias=False, indice_key=indice_key)

def conv3x3x1(in_planes, out_planes, stride=1, indice_key=None):
    indice_key = f"{indice_key}_3x3x1" if indice_key is not None else None
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(3, 3, 1), stride=stride,
                             padding=(1, 1, 0), bias=False, indice_key=indice_key)

def conv1x1x3(in_planes, out_planes, stride=1, indice_key=None):
    indice_key = f"{indice_key}_1x1x3" if indice_key is not None else None
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 1, 3), stride=stride,
                             padding=(0, 0, 1), bias=False, indice_key=indice_key)

def conv1x3x1(in_planes, out_planes, stride=1, indice_key=None):
    indice_key = f"{indice_key}_1x3x1" if indice_key is not None else None
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 3, 1), stride=stride,
                             padding=(0, 1, 0), bias=False, indice_key=indice_key)

def conv3x1x1(in_planes, out_planes, stride=1, indice_key=None):
    indice_key = f"{indice_key}_3x1x1" if indice_key is not None else None
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(3, 1, 1), stride=stride,
                             padding=(1, 0, 0), bias=False, indice_key=indice_key)

def conv1x1(in_planes, out_planes, stride=1, indice_key=None):
    indice_key = f"{indice_key}_1x1" if indice_key is not None else None
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=1, stride=stride,
                             padding=1, bias=False, indice_key=indice_key)