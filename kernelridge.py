from sklearn.kernel_ridge import KernelRidge

def train_kernelridge(X, y):
    model = KernelRidge(alpha=1.0, kernel="rbf")
    model.fit(X, y)
    return model
