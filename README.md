# IB-MHT: Information Bottleneck via Multiple Hypothesis Testing

This repository provides the implementation of the **IB-MHT algorithm** introduced in the paper [_Information Bottleneck via Multiple Hypothesis Testing (IB-MHT)_](https://arxiv.org/abs/2409.07325). This algorithm wraps around existing information bottleneck solvers and provides statistically valid guarantees on meeting the information-theoretic constraints. The IB-MHT method leverages Pareto testing and learn-then-test (LTT) frameworks to ensure statistical robustness.

## Overview

The Information Bottleneck (IB) framework is widely used in machine learning to extract compressed features that are informative for downstream tasks. While traditional approaches rely on heuristic tuning of hyperparameters without guarantees, IB-MHT offers a statistically valid solution to ensure that the learned features meet the IB constraints with high probability.

The algorithm, which builds on Pareto testing and the learn-then-test method, optimizes the mutual information between features and outputs while minimizing irrelevant information. The IB-MHT algorithm wraps around existing IB solvers to ensure that the extracted features meet predefined constraints on mutual information, even when the dataset size is limited.


## Data Preparation

To use the IB-MHT algorithm, you will need to prepare your data in the form of CSV files. These files should correspond to the input variables \( X \), output variables \( Y \), and the feature representations \( Z \).

### Data Format:
- **Rows**: Each row corresponds to a set of candidate hyperparameters.
- **Columns**: Each column represents a data point collected for each hyperparameter.

The repository includes example files used for the MNIST experiment from the paper. These can be found in the `examples` folder.

### Data Requirements:
- Prepare the `data_X.csv`, `data_Y.csv`, and `data_Z.csv` files before running the algorithm.
- Ensure that the number of rows matches the number of candidate hyperparameters, and the number of columns matches the number of data points.

## Usage

### Running the IB-MHT Algorithm:

1. Ensure that the `data_X.csv`, `data_Y.csv`, and `data_Z.csv` files are placed in the correct directory.
   
2. Modify the necessary parameters in the code:
   - You may need to adjust the values of the user-defined parameters, such as the threshold \( \alpha \) for the mutual information constraint, and the outage probability \( \delta \).


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

If you have any questions or need further information, feel free to reach out:

- **Email**: amirmohammad.farzaneh@kcl.ac.uk

## Citation

If you find this project useful for your research, please consider citing the following paper:

```bibtex
@misc{farzaneh2024statisticallyvalidinformationbottleneck,
      title={Statistically Valid Information Bottleneck via Multiple Hypothesis Testing}, 
      author={Amirmohammad Farzaneh and Osvaldo Simeone},
      year={2024},
      eprint={2409.07325},
      archivePrefix={arXiv},
      primaryClass={cs.IT},
      url={https://arxiv.org/abs/2409.07325}, 
}
```
