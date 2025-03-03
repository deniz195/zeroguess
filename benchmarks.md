# ZeroGuess Benchmarks

This document outlines the benchmarking methodology used to evaluate ZeroGuess's performance compared to traditional parameter estimation approaches. The benchmarks are designed to be reproducible, transparent, and representative of real-world curve fitting scenarios.

## Methodology

All benchmarks follow a consistent methodology:

1. **Parameter Sampling**: Random parameter sets are generated within predefined ranges
2. **Curve Fitting**: Each parameter set is used to fit curves using both ZeroGuess and traditional methods
3. **Evaluation**: The quality of fits is assessed by comparing converged parameters to true parameters
4. **Visualization**: For each parameter set, we generate visualizations showing:
   - The original curve (with true parameters)
   - The initial guess curve (with estimated parameters)
   - The fitted curve (with converged parameters)
   - A text box displaying numerical values for all parameters

## Benchmark Scenarios

### 1. Direct Comparison with lmfit

This benchmark directly compares ZeroGuess's parameter estimation capabilities against lmfit's built-in estimation methods.

**Configuration:**
- **Function**: Wavelet function from `zeroguess.functions`
- **Sample Size**: 10 different random parameter sets
- **Methods Compared**: 
  - lmfit's default parameter estimation
  - ZeroGuess-enhanced lmfit integration
- **Success Criteria**: Converged parameters within 5% of true parameters

**Output:**
- Comparison table showing successful vs. failed fits for both methods
- Visualization of fits for each parameter set
- Summary statistics on convergence rate and accuracy

### 2. Performance Across Function Types

This benchmark evaluates ZeroGuess's performance across different types of fitting functions with varying complexity.

**Configuration:**
- **Functions**: All functions in `zeroguess.functions` (Gaussian, Multi-peak Gaussian, Multimodal, Damped Sine, Linear, Sigmoid, Double Sigmoid, Wavelet)
- **Sample Size**: 100 different random parameter sets per function
- **Success Criteria**: Converged parameters within 5% of true parameters

**Output:**
- Comparison table showing successful vs. failed fits for each function type
- Visualization of representative fits for each function type
- Summary statistics on convergence rate and accuracy by function complexity

## Running the Benchmarks

The benchmarks can be run using the following command:

```bash
python scripts/run_benchmarks.py
```

Results will be saved to the `benchmark_results` directory, including:
- CSV files with raw benchmark data
- PNG images of fits
- HTML report with interactive visualizations and summary statistics

## Interpreting Results

The benchmark results should be interpreted considering:

1. **Convergence Rate**: The percentage of fits that successfully converge to the true parameters
2. **Accuracy**: How close the converged parameters are to the true parameters
3. **Robustness**: Performance across different function types and parameter ranges
4. **Efficiency**: Number of iterations and time required to converge

A successful benchmark will demonstrate that ZeroGuess provides more reliable initial parameter estimates, leading to higher convergence rates and more accurate fits compared to traditional methods.








