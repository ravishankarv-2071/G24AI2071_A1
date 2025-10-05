from misc import load_data, get_X_y, repeated_evaluate
from sklearn.kernel_ridge import KernelRidge

def make_model():
    return KernelRidge(alpha=1.0, kernel='rbf')

def main():
    # Load data
    df = load_data()
    X, y = get_X_y(df)

    # Train and evaluate model
    mean_mse, std_mse, mses = repeated_evaluate(
        make_model, X.values, y, n_repeats=5, test_size=0.2, random_seed=1
    )

    # Print results
    print(f"KernelRidge average MSE over 5 repeats: {mean_mse:.4f} ± {std_mse:.4f}")
    print("KernelRidge per-repeat MSEs:", [round(m,4) for m in mses])

if __name__ == "__main__":
    main()

