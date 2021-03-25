import sys
import importlib


# for loading previously trained model


def setup_legacy_env():
    importlib.import_module('lib.train')
    sys.modules['ltr'] = sys.modules['lib.train']
    importlib.import_module('lib.models')
    sys.modules['ltr.models'] = sys.modules['lib.models']
    for m in ('littleboy',):
        importlib.import_module('lib.models.' + m)
        sys.modules['ltr.models.' + m] = sys.modules['lib.models.' + m]


def cleanup_legacy_env():
    del_modules = []
    for m in sys.modules.keys():
        if m.startswith('ltr'):
            del_modules.append(m)
    for m in del_modules:
        del sys.modules[m]