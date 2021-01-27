import configparser
import collections

def is_float(val):
    try:
        num = float(val)
    except ValueError:
        return False
    return True

def is_int(val):
    try:
        num = int(val)
    except ValueError:
        return False
    return True

def parse_config(config_section, config_path):
    """
    Reads configuration from the file and returns a dictionary.
    """
    config_parser = configparser.SafeConfigParser(allow_no_value=True)
    config_parser.read(config_path)
    config = collections.OrderedDict()
    for key, value in config_parser.items(config_section):
        if value is None or len(value.strip()) == 0:
            config[key] = None
        elif value.lower() in ["true", "false"]:
            config[key] = config_parser.getboolean(config_section, key)
        elif is_int(value):
            config[key] = config_parser.getint(config_section, key)
        elif is_float(value):
            config[key] = config_parser.getfloat(config_section, key)
        else:
            config[key] = config_parser.get(config_section, key)
    return config

def print_config(config):
    print("######## Config ########")
    for key, value in config.items():
        print("{}: {}".format(key, value))
    return


def adjust_lr(optimizer, step, lr0, warmup):
    """to adjust the learning rate"""
    step = step + 1 # plus 1 to avoid ZeroDivisionError
    lr = lr0 * min(step**(-0.5), step*(warmup**(-1.5)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return
