'''
Надо это влить в агента после настройки просмотра самих статей
'''

import subprocess
import os

def try_tex(tex_file: str) -> str:
    tex_path = os.path.abspath(tex_file)
    tex_dir = os.path.dirname(tex_path) or "."

    for _ in range(2):
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", tex_dir, tex_file],
            cwd=tex_dir,
            capture_output=True,
            text=True
            )
        if result.returncode != 0:
            return "Ошибка компиляции:" + result.stdout[-2000:]

    pdf_path = os.path.join(tex_dir, os.path.splitext(os.path.basename(tex_path))[0] + ".pdf")
    return f"Готово:{pdf_path}"
