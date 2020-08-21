python -m venv venv
call venv/Scripts/activate.bat
pip install pylint
pip install mypy
pip install flake8
pip install black 
pip install pytest
pip install -e .
code .