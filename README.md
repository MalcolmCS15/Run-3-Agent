# Run-3-Agent
How to start and get ur env up to date/work with the notebook:

### Prerequisites
- Python 3.11 installed

### Installation Steps

1. Clone the repository:
git clone it in ur csci1470 foldre

2. Create a virtual environment:
python3.11 -m venv run3_env (make sure ur in the final-run3 folder)


3. Activate the virtual environment:
source run3_env/bin/activate
(this is macOS idk windows)

4. Install dependencies:
pip install -r requirements.txt
(I have pre installed a bunch of stuff, if down the line u need to install something then tell everyone)


5. Install Jupyter kernel:
pip install jupyter notebook ipykernel
python -m ipykernel install --user --name=run3_env --display-name="Python 3.11 (Run3 Project)"
(this line will take our "run3_env" venv and make it a kernel for a notebook, meaning we can run a notebook with it)

6. Launch Jupyter:
jupyter notebook
then in the page open the run3.ipynb file (this is our notebook)


7. Select the kernel: **Kernel** → **Change kernel** → **Python 3.11 (Run3 Project)**


## Deactivate Environment
when ur done or want to get out of the venv run "deactivate" in the terminal
make sure to git push and all that stuff. Make sure to "save" notebook before pushing it
