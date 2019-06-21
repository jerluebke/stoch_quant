# FP Stochastic Quantization

---

_Aim: Reproduce a quantum mechanical transition amplitude with a Langevin
equation using the method of stochastic quantization_

---

## Usage

__Files__ (in `langevin_simulation/`):
 * `langevin_solver.py`: python class for the Langevin simulation
 * `potentials.py`: several predefined potentials to use with the simulation
 * `config.py`: predefined simulation and plotting parameters
 * `plot_it.py`: functions to animate the results

__Quick start__:
In `plot_it.py` choose a potential and which quantities to display (see `if
__name__` block). If desired, adjust potential parameters in `config.py`.

To start the simulation, do:
```
python3 ./plot_it.py
```


>  vim: set ff=unix tw=79 sw=4 ts=4 et ic ai : 
