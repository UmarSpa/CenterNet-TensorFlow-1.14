import os
import yaml

def init_cfg(args):
    configs = merge_yaml(args.cfg_file)

    sys_cfg = configs['SYSTEM']
    db_cfg = configs['DATASET']
    model_cfg = configs['MODEL']
    del configs, args

    if sys_cfg['debug_flag']:
        sys_cfg['gpus'] = '1'
        sys_cfg['batch_size'] = 2

    os.environ["CUDA_VISIBLE_DEVICES"] = sys_cfg['gpus']
    sys_cfg['NUM_GPUS'] = len([int(x) for x in sys_cfg['gpus'].split(',')])
    if sys_cfg['NUM_GPUS'] > 1:
        sys_cfg['multi_gpu'] = True
    else:
        sys_cfg['multi_gpu'] = False

    db_cfg['backbone_type'] = model_cfg['backbone_type']
    return sys_cfg, db_cfg, model_cfg


def get_all_hierarchy(yaml_fn):
    with open(yaml_fn, 'r') as cfile:
        child = yaml.load(cfile, Loader=yaml.SafeLoader)
        if "__PARENT__" in child.keys():
            return get_all_hierarchy(child["__PARENT__"]) + [yaml_fn]
        else:
            return [yaml_fn]


def merge_dict(source, destination):
    for key, value in source.items():
        if isinstance(value, dict):
            node = destination.setdefault(key, {})
            merge_dict(value, node)
        else:
            destination[key] = value

    return destination


def merge_yaml(yaml_fn):
    cfg_fns = get_all_hierarchy(yaml_fn)
    res = {}
    for cfg_fn in cfg_fns:
        with open(cfg_fn, 'r') as cfile:
            cfg_dict = yaml.load(cfile, Loader=yaml.SafeLoader)
            res = merge_dict(cfg_dict, res)
    return res
