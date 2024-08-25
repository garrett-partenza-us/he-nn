# HE-based General Matrix Multiplication (HEGMM)

This repository contains my personal implementation of algorithms presented in the paper **"Secure and Efficient General Matrix Multiplication On Cloud Using Homomorphic Encryption"** by Yang Gao, Gang Quan, Soamar Homsi, Wujie Wen, and Liqiang Wang.

## Abstract

Despite the enormous technical and financial advantages of cloud computing, security and privacy have always been the primary concerns for adopting cloud computing facilities, especially for government agencies and commercial sectors with high-security requirements. Homomorphic Encryption (HE) has recently emerged as an effective tool in ensuring privacy and security for sensitive applications by allowing computing on encrypted data. One major obstacle to employing HE-based computation, however, is its excessive computational cost, which can be orders of magnitude higher than its counterpart based on plaintext. In this paper, we study the problem of how to reduce the HE-based computational cost for general Matrix Multiplication (MM), i.e., a fundamental building block for numerous practical applications, by taking advantage of the Single Instruction Multiple Data (SIMD) operations supported by HE schemes. Specifically, we develop a novel element-wise algorithm for general matrix multiplication, based on which we propose two HE-based General Matrix Multiplication (HEGMM) algorithms to reduce the HE computation cost. Our experimental results show that our algorithms can significantly outperform the state-of-the-art approaches of HE-based matrix multiplication.

## Installation 

Prior to installation, please ensure you have installed the Python SEAL bindings
from the Python SEAL repository. To set up the environment for running the code, follow these steps: 

1. **Clone the repository:** 
```
git clone https://github.com/garrett-partenza-us/hegmm.git
cd hegmm
```
2. **Create and activate a virtual environment**
```
python3 -m venv venv 
source venv/bin/activate
pip install -r requirements.txt
```
3. **Run test cases**
```
python hegmm.py
python hegmm_linear_transformation.py
```
