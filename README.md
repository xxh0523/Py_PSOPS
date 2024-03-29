# Python API for PSOPS
Py_PSOPS is the Python API for PSOPS. 

PSOPS stands for Power System Optimal Parameter Selection. It is a cpp-writen power system electromechanical simulator. 

Currently, It can solve power flow with Newton-Raphson method and perform electromechanical simulation with the implicit trapezoidal method. More details can be found in the **References**.

The cpp PSOPS is compiled as a library file. The Python ctypes library is used to access the external functions of PSOPS. Only the Python API is made public. 

Currently, the trained torch modules (see **referenc [2]**) are not be supported when using library files PSOPS_Source.dll or PSOPS.so.1.0.0. The neural modules can only be supported in excution files PSOPS_Source.exe or PSOPS_Source. If you want the excution files PSOPS_Source.exe or PSOPS_Source, please contact us (eexiaoxh@gmail.com). Also, see https://github.com/xxh0523/Py_PSNODE for more details.

# Environment

- **Windows: 7, 8, 10, 11**

- **Linux: Ubuntu 18.04, Ubuntu 20.04**

- **Python 3.6, 3.7, 3.8, 3.9, 3.10**

- NOTE: If encounter with **FileNotFoundError** for `ctypes.cdll.LoadLibrary()`, install **qtwebkit**. The conda command is as follows.

```
conda install qtwebkit
```

# Initialization
1.  Clone / Pull the codes.

# Directory
## dll_linux
When using Py_PSOPS on Linux platform, all the .so files needed for PSOPS are here. They can be downloaded on [GitHub](https://github.com/xxh0523/Py_PSOPS/releases).

## dll_win
When using Py_PSOPS on Windows platform, all the .dll files needed for PSOPS are here. They can be downloaded on [GitHub](https://github.com/xxh0523/Py_PSOPS/releases).

## sample
The original computation files needed for simulation are here. Test cases of IEEE-9, IEEE-39, and IEEE-118 are given. The dynamic component models are built based on the Power System Analysis Software Package (PSASP) V6.28 of the China Electric Power Research Institute (CEPRI).

## config.txt
It is the configuration file for PSOPS. The settings include sample location, data version, node ordering method, function setting, etc. 

## py_psops.py
It contains all the Python API. The codes in py_psops.py can be divided into calculation information, bus information, acline information, transformer information, generator information, load information, network information, fault and disturbance information, and integrated function. Details are in the comments. 

## sample_generators.py & generate_samples.py
An exmple of using Py_PSOPS to generate simulation samples.

# Debug and Run
After putting all the files in place, we can debug and run Py_PSOPS normally.

## Q&A
1.  **FileNotFoundError** for `ctypes.cdll.LoadLibrary()` 

    Try: install **qtwebkit**, e.g., `conda install qtwebkit`. Change command accordingly. 

1.  OSError: libQt5Core.so.5: cannot open shared object file: No such file or directory

    Try: `sudo apt install libqt5core5a`. Change command accordingly.

2.  OSError: libomp.so.5: cannot open shared object file: No such file or directory

    Try: `sudo apt install libomp5`. Change command accordingly.

3.  AssertionError: read settings failure!

    This assertation means there are something wrong with the **config.txt** file.
    
    The most common mistake is setting a wrong directory of the data file.

    Try: Edit the **config.txt**. Change the line starting with **Dir** and set a right directory of the data file.

4.  [WinError 127], The specified procedure could not be found.

    This error is caused by the usage of an incorrect Qt5Core.dll.

    Try: Replace the **Qt5Core.dll** in **'./dll_win'** with the **Qt5Core.dll** in **'\$conda_root\$\envs\\\$your_env\$\Library\bin'**.

    If you do not know which Qt5Core.dll should be used, you can install Qt5.2.1 and run **windeploy.exe PSOPS_Source.dll** to find the right location of Qt5Core.dll. 

# TODO
1. Develop more external functions of PSOPS, e.g., changing the calculation file location, changing the node ordering method, changing the sparse vector method, etc.

2. Manage MPI supportability of the Python API. 

3. We will publish other applications such as neural ODE and reinforcement learning in the close future.

# References
[1] **T. Xiao**, Y. Chen*, J. Wang, S. Huang, W. Tong, and T. He, “Exploration of AI-Oriented Power System Transient Stability Simulations,” *Journal of Modern Power Systems and Clean Energy*, vol. 11, no. 2, pp. 401–411, Mar. 2023, doi: [10.35833/MPCE.2022.000099](https://ieeexplore.ieee.org/document/9833418), [arxiv](http://arxiv.org/abs/2110.00931).


[2] **T. Xiao**, Y. Chen*, T. He, and H. Guan, “Feasibility Study of Neural ODE and DAE Modules for Power System Dynamic Component Modeling,” *IEEE Transactions on Power Systems*, vol. 38, no. 3, pp. 2666–2678, May 2023, doi: [10.1109/TPWRS.2022.3194570](https://ieeexplore.ieee.org/document/9844253), [arxiv](https://arxiv.org/abs/2110.12981).

[3] **T. Xiao**, W. Tong, and J. Wang*, “A New Fully Parallel BBDF Method in Transient Stability Simulations,” *IEEE Trans. Power Syst.*, vol. 35, no. 1, pp. 304–314, Jan. 2020, doi: [10.1109/TPWRS.2019.2933637](https://ieeexplore.ieee.org/document/8798601/).

[4] **T. Xiao**, W. Tong, and J. Wang*, “Study on Reducing the Parallel Overhead of the BBDF Method for Power System Transient Stability Simulations,” *IEEE Trans. Power Syst.*, vol. 35, no. 1, pp. 539–550, Jan. 2020, doi: [10.1109/TPWRS.2019.2929775](https://ieeexplore.ieee.org/document/8765766/).

[5] **T. Xiao**, J. Wang*, Y. Gao, and D. Gan, “Improved Sparsity Techniques for Solving Network Equations in Transient Stability Simulations,” *IEEE Trans. Power Syst.*, vol. 33, no. 5, pp. 4878–4888, Sep. 2018, doi: [10.1109/TPWRS.2018.2803200](https://ieeexplore.ieee.org/document/8283798/).
