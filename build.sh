#!/bin/bash
set -e
pip install --no-cache-dir -r requirements.txt
pip install --upgrade pip
python -c "import surprise; print('surprise安装成功')"
