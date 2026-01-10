#!/usr/bin/env python3
"""
Ejecutor directo para update_git_repo.py - ejecuta directamente sin shell
"""
import os
import sys
sys.path.insert(0, '/home/zarkentroska/Documentos/adas3')
os.chdir('/home/zarkentroska/Documentos/adas3')

# Ejecutar el script directamente
with open('update_git_repo.py', 'r', encoding='utf-8') as f:
    code = f.read()
    exec(compile(code, 'update_git_repo.py', 'exec'))


