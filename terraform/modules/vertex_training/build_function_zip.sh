#!/bin/bash
# budget_enforcer.zip を生成するスクリプト
# terraform apply の前に実行する
set -e
cd "$(dirname "$0")/budget_enforcer"
zip -j ../budget_enforcer.zip main.py requirements.txt
echo "budget_enforcer.zip created"
