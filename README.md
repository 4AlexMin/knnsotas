# KNN-based SOTAs Implementation

This repository contains implementations of several KNN-based State-of-the-Art (SOTA) algorithms. Feel free to add them to your benchmarks. 

These algorithms have been shown to perform well in various classification tasks, and can be catogerized as Adaptive-kNN and Graph-based kNN (kNNG).

## Algorithms Included

1. Adaptive-kNN
   1. SMKNN [1]
   2. LMKNN [1]
   3. PL-kNN [2] [*](https://www.softwareimpacts.com/article/S2665-9638(22)00143-9/fulltext)
   4. LV-kNN [3]
2. Graph-based kNN
   1. Centered kNNG [4]
   2. AKNNG [5]
   3. MAKNNG [5]
   4. Mutual kNNG
   5. Plain kNNG

## Usage
- You can directly perform any algorithm as a sklearn classifier (i.e., `fit()` and `predict()`)

- Or you can acquire all algorithms as:
   ```python
   from sota.sotas import get_sota_models
   models = get_sota_models()
   ```


## Contributing

We welcome contributions from the community. If you find any issues or have ideas for improvements, please open an issue or create a pull request.

Special thanks to Mr. Cai (leading author in [5]), who provide us with the MatLab code of AKNNG and MAKNNG. 

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References
[1]: S. M. Ayyad, A. I. Saleh, and L. M. Labib, “Gene expression cancer classification using modified K-Nearest Neighbors technique,” Biosystems, vol. 176, pp. 41–51, Feb. 2019, doi: 10.1016/j.biosystems.2018.12.009.

[2]: D. S. Jodas, L. A. Passos, A. Adeel, and J. P. Papa, “PL-k NN: A Parameterless Nearest Neighbors Classifier,” in 2022 29th International Conference on Systems, Signals and Image Processing (IWSSIP), Sofia, Bulgaria: IEEE, Jun. 2022, pp. 1–4. doi: 10.1109/IWSSIP55020.2022.9854445.

[3]: N. Garcia-Pedrajas, J. A. Romero Del Castillo, and G. Cerruela-Garcia, “A Proposal for Local $k$ Values for $k$-Nearest Neighbor Rule,” IEEE Transactions on Neural Networks and Learning Systems, vol. 28, no. 2, pp. 470–475, 2017, doi: 10.1109/TNNLS.2015.2506821.

[4]: I. Suzuki and K. Hara, “Centered kNN Graph for Semi-Supervised Learning,” in Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval, Shinjuku Tokyo Japan: ACM, Aug. 2017, pp. 857–860. doi: 10.1145/3077136.3080662.

[5] Y. Cai, J. Z. Huang, and J. Yin, “A new method to build the adaptive k-nearest neighbors similarity graph matrix for spectral clustering,” Neurocomputing, vol. 493, pp. 191–203, Jul. 2022, doi: 10.1016/j.neucom.2022.04.030.
