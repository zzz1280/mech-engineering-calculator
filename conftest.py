"""pytest 配置：让 tests/ 下的文件能直接 import 项目根目录的 calcs 包。

有了这个文件，无论在项目根目录还是 tests/ 目录执行 pytest，都能找到 calcs。
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
