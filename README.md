# Python API for PSOPS
Py_PSOPS is the Python API for PSOPS. PSOPS stands for Power System Optimal Parameter Selection.

This is the implementation of the following work.

[1]T. Xiao, Y. Chen, J. Wang, S. Huang, W. Tong, and T. He, “Exploration of AI-Oriented Power System Transient Stability Simulations,” [Online]. Available: http://arxiv.org/abs/2110.00931.

Compatible with both windows and linux. To download and use, please refer to **RELEASE** page.

# Installation
1.  Pull the codes.

2.  Download relase files according to the operating system, unzip the files, and put the unzipped folder with the source codes.

# Docs
1.  The details are in the comments of the py_psops.py.

2.  The source codes in py_psops.py can be divided into calculation information, bus information, acline information, transformer information, generator information, load information, network information, fault and disturbance information, and integrated function.

# Q&A
1.  OSError: libQt5Core.so.5: cannot open shared object file: No such file or directory

    Try： **sudo apt install libqt5core5a**，change command according to the Operating System.

2.  OSError: libomp.so.5: cannot open shared object file: No such file or directory

    Try:  **sudo apt install libomp5**， change command according to the Operating System.

3.  AssertionError: read settings failure!

    This assertation means there are something wrong with the **config.txt** file.
    
    The most common mistake is setting a wrong directory of the data file.

    Try: Edit the **config.txt**. Change the line starting with **Dir** and set a right directory of the data file.

