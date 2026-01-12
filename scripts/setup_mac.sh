python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
pip install tensorflow-metal

python -m ipykernel install --user --name ari3205 --display-name "ARI3205 (.venv)"