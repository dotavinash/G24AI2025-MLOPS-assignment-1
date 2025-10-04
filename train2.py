from sklearn.kernel_ridge import KernelRidge
from misc import load_data, split_data, build_pipeline, train_and_eval

def main():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=42)
    model = KernelRidge(alpha=1.0, kernel="rbf")
    pipe = build_pipeline(model, scale=True)  # scaling helps kernel methods
    mse, _ = train_and_eval(pipe, X_train, y_train, X_test, y_test)
    print(f"[KernelRidge] Test MSE: {mse:.4f}")

if __name__ == "__main__":
    main()
