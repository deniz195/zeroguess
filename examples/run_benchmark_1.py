#!/usr/bin/env python
"""
Script to run benchmarks for ZeroGuess.

Usage:
    python scripts/run_benchmarks.py [benchmark_name]

Benchmark names:
    lmfit_comparison: Direct comparison with lmfit (default)
    function_types: Performance across function types
    all: Run all benchmarks
"""

import argparse
import time
import warnings
from pathlib import Path

import lmfit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from zeroguess.functions import DoubleGaussianFunction, DoubleSigmoidFunction, WaveletFunction, add_gaussian_noise
from zeroguess.integration import ZeroGuessModel

# Create output directory
BENCHMARK_RESULTS_DIR = Path(__file__).parent / "benchmark_results"
BENCHMARK_RESULTS_DIR.mkdir(exist_ok=True)

DEFAULT_TOLERANCE = 0.10


def evaluate_fit_quality(true_params, fitted_params, tolerance=DEFAULT_TOLERANCE):
    """Evaluate the quality of a fit by comparing parameters.

    Args:
        true_params: dictionary of true parameter values
        fitted_params: dictionary of fitted parameter values
        tolerance: relative tolerance for parameter values

    Returns:
        success: whether the fit was successful
        relative_errors: dictionary of relative errors for each parameter
    """
    success = True
    relative_errors = {}

    for param_name, true_value in true_params.items():
        fitted_value = fitted_params[param_name]

        # Handle phase parameter specially (circular parameter)
        if param_name == "phase":
            # Normalize phases to [0, 2π]
            true_phase = true_value % (2 * np.pi)
            fitted_phase = fitted_value % (2 * np.pi)

            # Calculate minimum circular distance
            diff = min(
                abs(fitted_phase - true_phase),
                abs(fitted_phase - true_phase + 2 * np.pi),
                abs(fitted_phase - true_phase - 2 * np.pi),
            )
            relative_error = diff / (2 * np.pi)
        else:
            # For other parameters, calculate regular relative error
            relative_error = abs(fitted_value - true_value) / max(abs(true_value), 1e-10)

        relative_errors[param_name] = relative_error

        # Check if parameter is within tolerance
        if relative_error > tolerance:
            success = False

    return success, relative_errors


def visualize_fit(
    output_dir,
    fit_func,
    x_data,
    y_data,
    y_true,
    true_params,
    initial_params,
    fitted_params,
    method_name,
    param_set_idx,
    success,
    relative_errors,
    tolerance=DEFAULT_TOLERANCE,
):
    """Create visualization of the fit results.

    Args:
        x_data: x values for the data
        y_data: noisy y values
        y_true: true y values without noise
        true_params: dictionary of true parameter values
        initial_params: dictionary of initial parameter values
        fitted_params: dictionary of fitted parameter values
        method_name: name of the fitting method
        param_set_idx: index of the parameter set
        success: whether the fit was successful
        relative_errors: dictionary of relative errors for each parameter
    """

    # Calculate curves
    y_initial = fit_func(x_data, **initial_params)
    y_fitted = fit_func(x_data, **fitted_params)

    # Create figure
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(3, 1, height_ratios=[3, 1, 1])

    # Plot data and fits
    ax_main = fig.add_subplot(gs[0, 0])
    ax_main.plot(x_data, y_data, "ko", alpha=0.5, label="Noisy Data")
    ax_main.plot(x_data, y_true, "b-", label="True Curve")
    ax_main.plot(x_data, y_initial, "g--", label="Initial Guess")
    ax_main.plot(x_data, y_fitted, "r-", label="Fitted Curve")
    ax_main.set_xlabel("x")
    ax_main.set_ylabel("y")
    ax_main.legend()
    ax_main.set_title(f"Wavelet Fit - {method_name} - Set {param_set_idx+1}")

    # Add success/failure indicator
    if success:
        status_text = "SUCCESS"
        status_color = "green"
    else:
        status_text = "FAILURE"
        status_color = "red"
    ax_main.text(
        0.05,
        0.95,
        status_text,
        transform=ax_main.transAxes,
        fontsize=14,
        fontweight="bold",
        color=status_color,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Create parameter table
    ax_table = fig.add_subplot(gs[2, 0])
    ax_table.axis("off")

    # Prepare table data
    param_names = list(true_params.keys())
    table_data = []

    # Header row
    table_data.append(["Parameter", "True", "Initial", "Fitted", "Rel. Error"])

    # Parameter rows
    for param_name in param_names:
        true_val = true_params[param_name]
        initial_val = initial_params[param_name]
        fitted_val = fitted_params[param_name]
        rel_error = relative_errors[param_name]

        # Format values
        if param_name == "phase":
            # Format phase in radians
            true_str = f"{true_val:.2f}"
            initial_str = f"{initial_val:.2f}"
            fitted_str = f"{fitted_val:.2f}"
        else:
            true_str = f"{true_val:.4f}"
            initial_str = f"{initial_val:.4f}"
            fitted_str = f"{fitted_val:.4f}"

        # Format error with color
        if rel_error <= tolerance:
            error_str = f"{rel_error:.2%}"
        else:
            error_str = f"{rel_error:.2%}"

        table_data.append([param_name, true_str, initial_str, fitted_str, error_str])

    # Create the table
    table = ax_table.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Color the error cells
    for i, param_name in enumerate(param_names):
        rel_error = relative_errors[param_name]
        if rel_error <= tolerance:
            table[(i + 1, 4)].set_facecolor("#d8f3dc")  # light green
        else:
            table[(i + 1, 4)].set_facecolor("#ffccd5")  # light red

    # Add residuals plot
    ax_resid = fig.add_subplot(gs[1, 0])
    ax_resid.plot(x_data, y_data - y_fitted, "ko", alpha=0.5)
    ax_resid.axhline(y=0, color="r", linestyle="-", alpha=0.3)
    ax_resid.set_xlabel("x")
    ax_resid.set_ylabel("Residuals")

    # Adjust layout and save
    plt.tight_layout()
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / f"{method_name.replace(' ', '_').lower()}_set_{param_set_idx+1}.png", dpi=150)
    plt.close()


def run_lmfit_comparison_benchmark(  # noqa: C901
    function_name, n_samples=50, noise_level=0.05, tolerance=DEFAULT_TOLERANCE
):
    """Run benchmark comparing ZeroGuess with lmfit.

    Args:
        n_samples: number of parameter sets to test
        noise_level: relative noise level for generated data

    Returns:
        results_df: DataFrame with benchmark results
    """

    print(f"\n=== Running Direct Comparison with lmfit Benchmark for {function_name} ===\n")

    # Get parameter ranges from fit function
    if function_name == "double_gaussian":
        fit_func = DoubleGaussianFunction()
    elif function_name == "wavelet":
        fit_func = WaveletFunction()
    elif function_name == "double_sigmoid":
        fit_func = DoubleSigmoidFunction()
    else:
        raise ValueError(f"Invalid function name: {function_name}")

    assert fit_func.__name__
    param_ranges = fit_func.param_ranges

    # Set up output directory
    output_dir = BENCHMARK_RESULTS_DIR / "lmfit_comparison" / fit_func.__name__
    output_dir.mkdir(exist_ok=True, parents=True)

    # Set up x data
    x_data = fit_func.default_independent_vars["x"]

    # Create models with or without ZeroGuess integration
    model_lmfit = lmfit.Model(fit_func)

    model_zg = ZeroGuessModel(
        fit_func,
        independent_vars_sampling={"x": x_data},
        estimator_settings={
            "make_canonical": fit_func.get_canonical_params,
            "add_noise": True,
            "noise_level": noise_level,
            "snapshot_path": output_dir / f"benchmark_estimator_{function_name}.pth",
        },
    )

    # Set parameter bounds
    params_origin = {}
    for param_name, (min_val, max_val) in param_ranges.items():
        model_zg.set_param_hint(param_name, min=min_val, max=max_val)
        model_lmfit.set_param_hint(param_name, min=min_val, max=max_val)
        params_origin[param_name] = (max_val + min_val) / 2

    # Generate random parameter sets
    np.random.seed(42)  # For reproducibility
    param_sets = [fit_func.get_random_params(canonical=True) for _ in range(n_samples)]

    # Run comparison with ZeroGuess

    print("Training ZeroGuess estimator...")
    model_zg.zeroguess_train(n_epochs=500, n_samples=5000, device="cpu")

    lmfit_methods = ["least_squares", "dual_annealing"]

    # Initialize results storage
    results = []

    # Run benchmark for each parameter set
    for i, true_params in enumerate(param_sets):
        print(f"Processing parameter set {i+1}/{n_samples}...")

        # Generate noisy data
        y_true = fit_func(x_data, **true_params)
        y_data = add_gaussian_noise(y_true, sigma=noise_level, relative=False)

        # Test all methods
        base_methods = [
            ("Simple + lmfit", model_lmfit, "origin"),
            ("ZeroGuess + lmfit", model_zg, "zeroguess"),
            ("True + lmfit", model_lmfit, "trueguess"),
        ]
        all_methods = [
            (m[0] + f" ({lmfit_method})", m[1], lmfit_method, m[2])
            for lmfit_method in lmfit_methods
            for m in base_methods
        ]

        for method_name, model, lmfit_method, guess_method in all_methods:
            print(f"  Testing {method_name}...")

            # Time the parameter estimation
            start_time = time.time()

            # Get initial parameter estimates
            if guess_method == "zeroguess":
                initial_params = model.guess(y_data, x=x_data)
            elif guess_method == "trueguess":
                initial_params = model.make_params(**true_params)
            elif guess_method == "origin":
                initial_params = model.make_params(**params_origin)
            else:
                raise ValueError(f"Invalid guess method: {guess_method}")

            # Record estimation time
            estimation_time = time.time() - start_time

            # Extract initial parameter values
            initial_param_values = {name: param.value for name, param in initial_params.items()}

            # Fit the model
            start_time = time.time()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit_result = model.fit(y_data, initial_params, x=x_data, method=lmfit_method)
                fit_success = fit_result.success
                fit_message = fit_result.message
                fit_nfev = fit_result.nfev
                fitted_params = {name: param.value for name, param in fit_result.params.items()}
            except Exception as e:
                fit_success = False
                fit_message = str(e)
                fit_nfev = 0
                fitted_params = initial_param_values.copy()

            # Make parameters canonical
            fitted_params = fit_func.get_canonical_params(fitted_params)

            # Record fitting time
            fitting_time = time.time() - start_time

            # Evaluate fit quality
            param_success, relative_errors = evaluate_fit_quality(true_params, fitted_params)

            # Create visualization
            visualize_fit(
                output_dir,
                fit_func,
                x_data,
                y_data,
                y_true,
                true_params,
                initial_param_values,
                fitted_params,
                method_name,
                i,
                param_success,
                relative_errors,
            )

            # Store results
            results.append(
                {
                    "param_set": i + 1,
                    "method": method_name,
                    "estimation_time": estimation_time,
                    "fitting_time": fitting_time,
                    "fit_success": fit_success,
                    "param_success": param_success,
                    "fit_message": fit_message,
                    "fit_nfev": fit_nfev,
                    **{f"true_{k}": v for k, v in true_params.items()},
                    **{f"initial_{k}": v for k, v in initial_param_values.items()},
                    **{f"fitted_{k}": v for k, v in fitted_params.items()},
                    **{f"error_{k}": v for k, v in relative_errors.items()},
                }
            )

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Save results to CSV
    results_df.to_csv(output_dir / "results.csv", index=False)

    # Generate summary report
    generate_lmfit_comparison_report(results_df, output_dir, function_name, noise_level, tolerance)

    return results_df


def generate_lmfit_comparison_report(results_df, output_dir, function_name, noise_level=0.05, tolerance=DEFAULT_TOLERANCE):  # noqa: C901
    """Generate a summary report for the lmfit comparison benchmark.

    Args:
        results_df: DataFrame with benchmark results
        output_dir: directory to save the report
        function_name: name of the function being benchmarked
        noise_level: relative noise level used in the benchmark
        tolerance: relative tolerance for parameter values
    """
    # Group by method and calculate success rates
    method_summary = (
        results_df.groupby("method")
        .agg(
            {
                "fit_success": "mean",
                "param_success": "mean",
                "estimation_time": "mean",
                "fitting_time": "mean",
                "fit_nfev": "mean",
            }
        )
        .sort_values(["param_success", "fit_success", "fitting_time"], ascending=[True, True, False])
    )

    # Calculate parameter-specific error statistics
    param_names = [col.split("_")[1] for col in results_df.columns if col.startswith("error_")]
    for param in param_names:
        method_summary[f"avg_error_{param}"] = results_df.groupby("method")[f"error_{param}"].mean()

    # Create summary figure
    plt.figure(figsize=(10, 6))

    # Plot success rates
    ax1 = plt.subplot(121)
    (100 * method_summary[["fit_success", "param_success"]]).plot(kind="bar", ax=ax1, color=["#4e79a7", "#f28e2b"])
    ax1.set_title("Success Rates [%]")
    ax1.set_ylabel("Success Rate [%]")
    ax1.set_ylim(0, 115)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")
    ax1.legend(loc="lower right")

    # for i, v in enumerate(method_summary["fit_success"]):
    #     ax1.text(i - 0.15, 100*(v + 0.02), f"{v:.0%}", color="black", fontweight="bold")

    for i, v in enumerate(method_summary["param_success"]):
        ax1.text(i + 0.10, 100*(v + 0.02), f"{v:.0%}", color="black", fontweight="bold")

    # Plot average parameter errors
    ax2 = plt.subplot(122)
    error_cols = [col for col in method_summary.columns if col.startswith("avg_error_")]
    error_df = method_summary[error_cols].copy()
    error_df.columns = [col.split("_")[2] for col in error_df.columns]
    error_df.plot(kind="bar", ax=ax2, color=["#59a14f", "#e15759", "#76b7b2", "#edc949"])
    ax2.set_title("Average Parameter Errors")
    ax2.set_ylabel("Relative Error")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_dir / "summary.png", dpi=150)
    plt.close()

    # Create HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ZeroGuess Benchmark (lmfit for {function_name} function)</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .success {{ color: green; }}
            .failure {{ color: red; }}
            .summary-container {{ display: flex; flex-wrap: wrap; }}
            .summary-section {{ flex: 1; min-width: 300px; margin-right: 20px; }}
            img {{ max-width: 100%; height: auto; margin-top: 20px; }}
            .benchmark-info {{ 
                background-color: #f8f8f8; 
                padding: 10px; 
                border-radius: 5px; 
                margin-bottom: 15px; 
                line-height: 1.3;
                font-size: 0.85em;
            }}
            .benchmark-info h2 {{ 
                font-size: 1.2em; 
                margin-top: 5px;
                margin-bottom: 8px;
            }}
            .benchmark-info h3 {{ 
                font-size: 1.05em; 
                margin-top: 8px;
                margin-bottom: 5px;
            }}
            .benchmark-info p, .benchmark-info li {{ 
                margin: 3px 0;
            }}
            .benchmark-info ol, .benchmark-info ul {{
                padding-left: 20px;
                margin: 5px 0;
            }}
        </style>
    </head>
    <body>
        <h1>ZeroGuess Benchmark (lmfit for {function_name} function)</h1>
        
        <div class="benchmark-info">
            <h2>How This Benchmark Works</h2>
            <p>This benchmark evaluates different parameter estimation and curve fitting approaches using synthetic data with known ground truth values.</p>
            
            <h3>Methodology:</h3>
            <ol>
                <li><strong>Test Data Generation:</strong> {results_df['param_set'].nunique()} different parameter sets are randomly generated for the {function_name} function. For each parameter set, synthetic data is created with {noise_level*100:.0f}% noise added.</li>
                <li><strong>Methods Compared:</strong>  
                    <ul>
                        <li><strong>Simple + lmfit:</strong> Uses central values from parameter ranges as starting points</li>
                        <li><strong>ZeroGuess + lmfit:</strong> Uses ZeroGuess's neural network to estimate starting parameters</li>
                        <li><strong>True + lmfit:</strong> Uses the true parameters as starting points (best case scenario)</li>
                    </ul>
                    Each method is tested with different optimization algorithms (least_squares, dual_annealing)
                </li>
                <li><strong>Metrics:</strong>
                    <ul>
                        <li><strong>Fit Success:</strong> Whether the fitting algorithm converged</li>
                        <li><strong>Parameter Success:</strong> Whether all recovered parameters are within {tolerance*100:.0f}% of true values</li>
                        <li><strong>Computation Time:</strong> Time for parameter estimation and fitting</li>
                        <li><strong>Function Evaluations:</strong> Number of function calls required</li>
                    </ul>
                </li>
            </ol>
            
        </div>

        <h2>Summary for {function_name} function</h2>
        
        <h3>Success and Performance Metrics</h3>
        <table>
            <tr>
                <th>Method</th>
                <th>Fit Success</th>
                <th>Parameter Success</th>
                <th>Estimation Time (s)</th>
                <th>Fitting Time (s)</th>
                <th>Function Evaluations</th>
            </tr>
    """  # noqa: F541

    for method, row in method_summary.iterrows():
        fit_class = "success" if row["fit_success"] >= 0.5 else "failure"
        param_class = "success" if row["param_success"] >= 0.5 else "failure"

        html_content += f"""
            <tr>
                <td>{method}</td>
                <td class="{fit_class}">{row['fit_success']:.1%}</td>
                <td class="{param_class}">{row['param_success']:.1%}</td>
                <td>{row['estimation_time']:.4f}</td>
                <td>{row['fitting_time']:.4f}</td>
                <td>{row['fit_nfev']:.1f}</td>
            </tr>
        """

    html_content += """
        </table>

        <h3>Parameter Error Summary</h3>
        <table>
            <tr>
                <th>Method</th>
    """

    for param in param_names:
        html_content += f"<th>{param}</th>"

    html_content += "</tr>"

    for method, row in method_summary.iterrows():
        html_content += f"""
            <tr>
                <td>{method}</td>
        """

        for param in param_names:
            error_val = row[f"avg_error_{param}"]
            error_class = "success" if error_val <= tolerance else "failure"
            html_content += f'<td class="{error_class}">{error_val:.1%}</td>'

        html_content += "</tr>"

    html_content += """
        </table>

        <h2>Visualization</h2>
        <img src="summary.png" alt="Summary Chart">

        <h2>Sample Fits</h2>
    """

    # Add sample images
    methods = results_df["method"].unique()
    for method in methods:
        method_file_prefix = method.replace(" ", "_").lower()
        html_content += f"""
        <h3>{method}</h3>
        <div style="display: flex; flex-wrap: wrap;">
        """

        # Add navigation links
        html_content += """
        <div style="width: 100%; margin-bottom: 10px;">
            Links to all fits: 
        """
        
        # Generate numbered links to all sample files for this method
        max_samples = min(results_df["param_set"].nunique(), 100)  # Limit to 100 links
        for i in range(1, max_samples + 1):
            img_path = f"{method_file_prefix}_set_{i}.png"
            html_content += f'<a href="{img_path}" target="_blank">[{i}]</a> '
        
        html_content += """
        </div>
        """

        # Show the first 3 examples as embedded images (keeping original behavior)
        for i in range(min(3, results_df["param_set"].nunique())):
            img_path = f"{method_file_prefix}_set_{i+1}.png"
            html_content += f"""
            <div style="margin: 10px;">
                <img src="{img_path}" alt="{method} - Set {i+1}" style="max-width: 400px;">
            </div>
            """

        html_content += "</div>"

    html_content += """
    </body>
    </html>
    """

    # Write HTML report
    with open(output_dir / "report.html", "w") as f:
        f.write(html_content)

    print(f"\nReport generated at {output_dir / 'report.html'}")


def main():
    """Main entry point for the benchmark script."""
    parser = argparse.ArgumentParser(description="Run benchmarks for ZeroGuess")
    parser.add_argument(
        "benchmark",
        choices=["lmfit_comparison", "function_types", "all"],
        default="lmfit_comparison",
        nargs="?",
        help="Benchmark to run",
    )
    parser.add_argument(
        "--function",
        choices=["double_sigmoid", "double_gaussian", "wavelet"],
        default="wavelet",
        help="Function to use for the benchmark",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=50,
        help="Number of samples to use for the benchmark",
    )
    args = parser.parse_args()

    if args.benchmark == "lmfit_comparison" or args.benchmark == "all":
        run_lmfit_comparison_benchmark(args.function, args.n_samples)

    if args.benchmark == "function_types" or args.benchmark == "all":
        print("\nFunction types benchmark not implemented yet.")

    print("\nBenchmarks completed. Results saved to:", BENCHMARK_RESULTS_DIR)


if __name__ == "__main__":
    main()
