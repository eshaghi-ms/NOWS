# NOWS: Neural Operator Warm Starts

This repository accompanies the paper:

**NOWS: Neural Operator Warm Starts for Accelerating Iterative Solvers**

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

## Repository status

The **code and data will be made publicly available after the journal review process is complete.**

👉 **Please ⭐ star this repository** to get notified automatically when the public release is published.

---

## Paper citation

If you use or refer to this work, please cite:

```bibtex
@article{Eshaghi2025NOWS,
  title={Neural Operator Warm Starts for Accelerating Iterative Solvers},
  author={Eshaghi, Mohammad Sadegh and Anitescu, Cosmin and Valizadeh, Navid and Wang, Yizheng and Zhuang, Xiaoying and Rabczuk, Timon},
  year={2025},
  journal={TBD}
}
