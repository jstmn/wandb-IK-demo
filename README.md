# wandb-IK-demo

A sample training workflow of a 4dof planar robot to show off how to use Weights and Biases


## Instillation

```
python3.8 -m pip install --user virtualenv
python3.8 -m venv venv
source venv/bin/activate
pip install wheel
pip install -r requirements.txt
git submodule init; git submodule update # Install git submodules
cd thirdparty/FrEIA && python setup.py develop && cd ../../ # Install thirdparty libraries
```

## Running a sweep

```
wandb sweep hparam_sweep.yaml
# wandb: Creating sweep from: hparam_sweep.yaml
# wandb: Created sweep with ID: 0difi2vn
# wandb: View sweep at: https://wandb.ai/jeremysmorgan/XYZ/sweeps/0difi2
# wandb: Run sweep agent with: wandb agent jeremysmorgan/XYZ/0difi2vn


wandb agent jeremysmorgan/wandb_demo/0difi2vn
```


