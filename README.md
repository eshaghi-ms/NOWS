# NOWS: Neural Operator Warm Starts

This repository accompanies the paper:

**NOWS: Neural Operator Warm Starts for Accelerating Iterative Solvers**
*Published in Computer Methods in Applied Mechanics and Engineering (CMAME)*

---

## Overview

**Neural Operator Warm Starts (NOWS)** is a hybrid framework that accelerates classical numerical PDE solvers — such as Conjugate Gradient (CG) and GMRES — by using *neural operators* (e.g., FNO, DeepONet, VINO) to generate high-quality initial guesses.  
This reduces solver iterations and overall runtime while keeping the solver’s **stability**, **robustness**, and **convergence guarantees** fully intact.

NOWS is **solver-agnostic** and **discretization-agnostic**, integrating seamlessly with:
- Finite Element Method (FEM)
- Finite Difference Method (FDM)
- Isogeometric Analysis (IGA)
- Finite Volume Method (FVM)

Across diverse PDEs (Poisson, Darcy, Burgers, elasticity, Navier–Stokes), NOWS consistently achieves **25–90% runtime reduction**, demonstrating strong generalization across geometries, resolutions, and physical systems.

---
## NOWS framework (high-level)

<p align="center">
  <img src="docs/figs/flowchart.png" width="700">
</p>

*Workflow concept:*  
Neural operator → produces an initial guess → classical solver only needs to do the *fine* iterations → same accuracy, much lower cost.

<p align="center">
  <img src="docs/figs/dynamic.png" width="700">
</p>

---

## Repository status

✅ **The paper has been published in CMAME!**

The code and data are now publicly available in this repository.

👉 **Please ⭐ star this repository** to stay updated with new features and improvements.

---

## Abstract (summary)

Partial differential equations (PDEs) are the foundation of computational science and engineering, yet high-fidelity simulations remain computationally demanding.  
NOWS introduces a hybrid paradigm that leverages neural operator predictions to warm-start traditional iterative solvers.  
The neural operator rapidly provides an accurate initial guess, and the classical solver completes convergence.  
This hybridization achieves substantial speed-ups (up to 90%), retains full accuracy, and is compatible with any existing solver infrastructure.  
By combining the rapid inference of learned operators with the rigor of numerical solvers, NOWS offers a practical and trustworthy path toward accelerating high-fidelity PDE simulations.

---

## Key features

- 🚀 **Acceleration:** Reduces iteration count and runtime by up to 90%.
- 🧠 **Physics-consistent learning:** Can use data-driven, physics-informed, or hybrid training.
- ⚙️ **Solver-agnostic:** Works with CG, GMRES, and other Krylov or multigrid methods.
- 🧩 **Discretization-independent:** Supports FEM, FDM, IGA, and FVM backends.
- 🗺️ **Generalization:** Robust to mesh refinement, geometry variation, and PDE type.
- 🧮 **No modification needed:** Integrates directly with existing solvers.

---

## Contact

For questions or collaboration inquiries:

- **Mohammad Sadegh Eshaghi** — eshaghi.khanghah@iop.uni-hannover.de  
- **Cosmin Anitescu** — cosmin.anitescu@uni-weimar.de  
- **Timon Rabczuk** — timon.rabczuk@uni-weimar.de  

---

## Acknowledgments

The authors acknowledge the support of:

- The **German Academic Exchange Service (DAAD)**
- The **Compute Servers of TU Ilmenau** for providing computational resources.
  
## Paper citation

If you use or refer to this work, please cite:

```bibtex
@article{Eshaghi2025NOWS,
  title={Neural Operator Warm Starts for Accelerating Iterative Solvers},
  author={Eshaghi, Mohammad Sadegh and Anitescu, Cosmin and Valizadeh, Navid and Wang, Yizheng and Zhuang, Xiaoying and Rabczuk, Timon},
  year={2025},
  journal={Computer Methods in Applied Mechanics and Engineering},
  publisher={Elsevier},
  note={arXiv preprint: https://doi.org/10.48550/arXiv.2511.02481}
}
