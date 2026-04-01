# openpan/registry.py
class Registry:
    def __init__(self, name):
        self._name = name
        self._module_dict = {}

    def register_module(self, name=None):
        """装饰器：用于将类注册到字典中"""
        def _register(cls):
            module_name = name if name is not None else cls.__name__
            self._module_dict[module_name] = cls
            return cls
        return _register

    def build(self, cfg, **kwargs):
        """根据配置字典实例化模块"""
        if not isinstance(cfg, dict) or 'type' not in cfg:
            raise ValueError(f"Config must be a dict with 'type', got {cfg}")
        
        cfg_copy = cfg.copy()
        module_type = cfg_copy.pop('type')
        
        if module_type not in self._module_dict:
            raise KeyError(f"Module {module_type} not found in {self._name} registry")
        
        module_cls = self._module_dict[module_type]
        # 合并额外的 kwargs
        cfg_copy.update(kwargs)
        return module_cls(**cfg_copy)

# 实例化四大全局注册器
BACKBONES = Registry('backbone')
NECKS = Registry('neck')
HEADS = Registry('head')
LOSSES = Registry('loss')
MODELS = Registry('model')
DATASETS = Registry('dataset')