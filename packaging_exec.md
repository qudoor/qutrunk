本教程介绍如何将python项目打包成可执行三方库，包括打包项目内部依赖模块/包，申明本项目依赖的其他三方库，申明项目执行入口等

## 什么是可执行三方库

可执行三方库是带有项目可执行入口的三方包。除此之外，与项目中依赖的其他三方库无区别。  
常用于在控制台执行的脚本包，pip, pytest等工具均采用此方式。  
*本教程使用  [poetry](https://python-poetry.org/docs/)管理项目依赖和构建，需提前安装好python和poetry，建议使用虚拟环境。  
若习惯使用其他工具，可参考对应工具的文档，构建可执行三方包*

## 创建项目([示例项目(待补充链接)](http://developer.queco.cn/))

#### 进入为项目创建的空白目录

```shell
> cd qutrunk_app  
```

#### 项目初始化

###### 1. 直接使用示例中的项目描述文件pyproject.toml，并根据项目情况修改

###### 2. 使用poetry交互式命令行

```shell

> poetry init  
This command will guide you through creating your pyproject.toml config.  
> Package name [qutrunk_app]:    # 使用默认值  
> Version [0.1.0]:   # 使用默认值
> Description []: # 使用默认值  
> Author [guomengjie <guomengjie@qudoor.cn>, n to skip]:  # 使用默认值或跳过均可
> License []:  # 使用默认值   
> Compatible Python versions [^3.10]:  # 需要的python版本  
> Would you like to define your main dependencies interactively? (yes/no) [yes] no  # 后续再定义  
> Would you like to define your development dependencies interactively? (yes/no) [yes] no # 后续再定义  
Generated file  

[tool.poetry]  
name = "qutrunk-app"  
version = "0.1.0"  
description = ""  
authors = ["guomengjie <guomengjie@qudoor.cn>"]  
readme = "README.md"  
packages = [{include = "qutrunk_app"}]  

[tool.poetry.dependencies]  
python = "^3.10"  

[build-system]
requires = ["poetry-core"]  
build-backend = "poetry.core.masonry.api"

> Do you confirm generation? (yes/no) [yes] yes
```

**注释readme行，或在项目根目录下建立README.md文件**

#### 编写代码，包括创建依赖包, 引入三方依赖等

#### 创建可执行文件(项目入口文件)

在源码根目录创建__main__.py文件，完成后的目录结构如下

```
.
├── pyproject.toml
└── qutrunk_app
    ├── __main__.py
    └── foo
        ├── __init__.py
        └── dep.py
```

## 打包

在项目根目录执行 poetry build，即可在dist目录下生成所需的包

本地执行测试

```shell
> cd dist
> pip install qutrunk_app-0.1.0-py3-none-any.whl
> python -m qutrunk_app
```

## 上传(待QuSaas和QuPot完成后补充)

## 执行(待QuSaas和QuPot完成后补充)

