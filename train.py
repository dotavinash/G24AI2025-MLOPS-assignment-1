from sklearn.tree import DecisionTreeRegressor
from misc import load_data, split_data, build_pipeline, train_and_eval

def main():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=42)
    model = DecisionTreeRegressor(random_state=42)
    pipe = build_pipeline(model, scale=False)  # trees don't need scaling
    mse, _ = train_and_eval(pipe, X_train, y_train, X_test, y_test)
    print(f"[DecisionTreeRegressor] Test MSE: {mse:.4f}")

if __name__ == "__main__":
    main()
