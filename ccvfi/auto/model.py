from typing import Any, Optional, Union
from pathlib import Path

import torch

from ccvfi.config import CONFIG_REGISTRY
from ccvfi.model import MODEL_REGISTRY
from ccvfi.type import BaseConfig, ConfigType, ArchType, ModelType


class AutoModel:
    @staticmethod
    def from_pretrained(
        pretrained_model_name: Union[ConfigType, str],
        device: Optional[torch.device] = None,
        fp16: bool = True,
        compile: bool = False,
        compile_backend: Optional[str] = None,
        model_dir: Optional[str] = None,
        gh_proxy: Optional[str] = None,
    ) -> Any:
        """
        Get a model instance from a pretrained model name or custom model path.

        :param pretrained_model_name: The name of the pretrained model (registered in CONFIG_REGISTRY) or direct path to a .pkl model file.
        :param device: inference device
        :param fp16: use fp16 precision or not
        :param compile: use torch.compile or not
        :param compile_backend: backend of torch.compile
        :param model_dir: The path to cache the downloaded model. Should be a full path. If None, use default cache path.
        :param gh_proxy: The proxy for downloading from github release. Example: https://github.abskoop.workers.dev/
        :return:
        """

        # Check if pretrained_model_name is a path to a custom model
        print(f"🔍 AutoModel.from_pretrained called with: {pretrained_model_name}")
        print(f"🔍 Type: {type(pretrained_model_name)}")
        
        is_custom_path = AutoModel._is_custom_model_path(pretrained_model_name)
        print(f"🔍 _is_custom_model_path returned: {is_custom_path}")
        
        if is_custom_path:
            print(f"✅ Detected as custom path, creating custom config...")
            config = AutoModel._create_custom_rife_config(pretrained_model_name)
        else:
            print(f"❌ Not detected as custom path, looking in CONFIG_REGISTRY...")
            config = CONFIG_REGISTRY.get(pretrained_model_name)
        
        return AutoModel.from_config(
            config=config,
            device=device,
            fp16=fp16,
            compile=compile,
            compile_backend=compile_backend,
            model_dir=model_dir,
            gh_proxy=gh_proxy,
        )

    @staticmethod
    def from_config(
        config: Union[BaseConfig, Any],
        device: Optional[torch.device] = None,
        fp16: bool = True,
        compile: bool = False,
        compile_backend: Optional[str] = None,
        model_dir: Optional[str] = None,
        gh_proxy: Optional[str] = None,
    ) -> Any:
        """
        Get a model instance from a config.

        :param config: The config object. It should be registered in CONFIG_REGISTRY.
        :param device: inference device
        :param fp16: use fp16 precision or not
        :param compile: use torch.compile or not
        :param compile_backend: backend of torch.compile
        :param model_dir: The path to cache the downloaded model. Should be a full path. If None, use default cache path.
        :param gh_proxy: The proxy for downloading from github release. Example: https://github.abskoop.workers.dev/
        :return:
        """

        model = MODEL_REGISTRY.get(config.model)
        model = model(
            config=config,
            device=device,
            fp16=fp16,
            compile=compile,
            compile_backend=compile_backend,
            model_dir=model_dir,
            gh_proxy=gh_proxy,
        )

        return model

    @staticmethod
    def register(obj: Optional[Any] = None, name: Optional[str] = None) -> Any:
        """
        Register the given object under the name `obj.__name__` or the given name.
        Can be used as either a decorator or not. See docstring of this class for usage.

        :param obj: The object to register. If None, this is being used as a decorator.
        :param name: The name to register the object under. If None, use `obj.__name__`.
        :return:
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class: Any) -> Any:
                _name = name
                if _name is None:
                    _name = func_or_class.__name__
                MODEL_REGISTRY.register(obj=func_or_class, name=_name)
                return func_or_class

            return deco

        # used as a function call
        if name is None:
            name = obj.__name__
        MODEL_REGISTRY.register(obj=obj, name=name)

    @staticmethod
    def _is_custom_model_path(pretrained_model_name: Union[ConfigType, str]) -> bool:
        """
        Check if the pretrained_model_name is a direct path to a .pkl model file.

        :param pretrained_model_name: The model name or path to check
        :return: True if it's a custom model path, False otherwise
        """
        print(f"🔍 _is_custom_model_path: checking {pretrained_model_name}")
        
        if isinstance(pretrained_model_name, ConfigType):
            print(f"🔍 Is ConfigType, returning False")
            return False
        
        # Check if it's a path to a .pkl file
        path = Path(pretrained_model_name)
        print(f"🔍 Path object: {path}")
        print(f"🔍 path.exists(): {path.exists()}")
        print(f"🔍 path.is_file(): {path.is_file()}")
        print(f"🔍 path.suffix: {path.suffix}")
        print(f"🔍 path.suffix.lower(): {path.suffix.lower()}")
        
        result = (path.exists() and 
                  path.is_file() and 
                  path.suffix.lower() == '.pkl')
        print(f"🔍 Final result: {result}")
        return result

    @staticmethod
    def _create_custom_rife_config(model_path: str) -> BaseConfig:
        """
        Create a custom RIFE config for a .pkl model file.

        :param model_path: Direct path to the .pkl model file
        :return: A BaseConfig instance for the custom model
        """
        path = Path(model_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if not path.is_file():
            raise ValueError(f"Path is not a file: {model_path}")
        
        if path.suffix.lower() != '.pkl':
            raise ValueError(f"File must have .pkl extension: {model_path}")
        
        # Create a custom config for RIFE model
        # Use the filename (without extension) as part of the config name
        config = BaseConfig(
            name=f"custom_rife_{path.stem}",
            path=str(path.absolute()),
            arch=ArchType.IFNET,
            model=ModelType.RIFE,
            in_frame_count=2  # Default for RIFE models
        )
        
        return config
