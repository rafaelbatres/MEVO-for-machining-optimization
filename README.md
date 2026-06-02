# Experimental Assessment of the Metamodel-Based Evolutionary Optimizer (MEVO) for Machining Optimization
An implementation of the Metamodel-Based Evolutionary Optimizer (MEVO) for data-efficient machining optimization. This repository contains the code and data for the benchmarking of  MEVO against traditional Response Surface Methodology (RSM) in a slot-milling case study.

## Why this repository?

A major challenge in industrial machining optimization is the high cost associated with physical experimentation. This project addresses that challenge through a surrogate-assisted evolutionary optimization framework capable of identifying optimal cutting conditions while operating under a strictly limited experimental budget.

The proposed approach combines machine learning and evolutionary optimization to iteratively improve process understanding and guide experimentation toward promising regions of the search space.

---

## Key Features

### Adaptive Learning

Unlike static Design of Experiments (DoE)-based approaches such as Response Surface Methodology (RSM), MEVO continuously retrains its surrogate model after each physical experiment, progressively improving predictive accuracy as new data become available.

### MLP Surrogate Model

The framework employs a Multi-Layer Perceptron (MLP) regressor to model the machining process. This surrogate is capable of capturing complex nonlinear relationships and synergistic interactions among process parameters that are often difficult to represent using traditional quadratic models.

---

## Experimental Setup and Data

The experimental dataset consists of slot-milling trials performed on **Aluminum 6061-T651**.

### Input Parameters

| Parameter | Description        |
| --------- | ------------------ |
| `ap`      | Axial depth of cut |
| `fz`      | Feed per tooth     |
| `vc`      | Cutting speed      |

### Target Responses

| Response | Description                    |
| -------- | ------------------------------ |
| `ASCE`   | Active Specific Cutting Energy |
| `Ra`     | Average Surface Roughness      |

### Optimization Objective

The optimization process seeks to maximize a **Composite Desirability Index (`D`)**, which aggregates the conflicting objectives of:

* Minimizing energy consumption (`ASCE`)
* Minimizing surface roughness (`Ra`)

This formulation enables the simultaneous consideration of sustainability and product quality during machining optimization.

---

## Repository Contents

```text
├── data/               # Experimental datasets
├── src/                # Source code for MEVO and MEPSO
├── results/            # Optimization results and analysis
└── README.md
```

---

## Citation

If you use this repository in your research, please cite:

```bibtex
@article{MEVO_Machining,
  title={Experimental Assessment of the Metamodel-Based Evolutionary Optimizer (MEVO) for Machining Optimization},
  author={Velázquez-López, Antonio and Batres, Rafael and Miranda-Valenzuela, José Carlos and Calderón-Nájera, Juan de Dios},
journal = {International Journal of Advanced Manufacturing Technology},
  year={2026}
}
```

---

## License

This project is distributed under the license specified in the repository.
