import argparse
import pickle
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Product
from dataloading import get_dataset_and_denormalizer_sklearn
from draw import plot_predictions_vs_ground_truth_vanilla
from utils import compute_all_metrics, write_csv
from paths import GPCFSavePath


parser = argparse.ArgumentParser()
parser.add_argument("plant_number", type=int, help="Power plant number to be used for training")
args = parser.parse_args()


save_path = GPCFSavePath(args.plant_number)
# Define the basic RBF kernel (squared exponential) for wind speed and air density
kernel_wind_speed = RBF(length_scale=1.0)  # Kernel for wind speed
kernel_air_density = RBF(length_scale=1.0)  # Kernel for air density

# Combine the kernels by multiplying them (representing their joint effect)
simple_composite_kernel = Product(kernel_wind_speed, kernel_air_density)

# Initialize the Gaussian Process Regressor with the composite kernel
model = GaussianProcessRegressor(kernel=simple_composite_kernel, alpha=1e-2)

_, Y_train, X_nwp_train, denormalizer = get_dataset_and_denormalizer_sklearn(args.plant_number, "train", str(save_path))
model.fit(X_nwp_train, Y_train)
with open(save_path.get_model_path(), "wb") as f:
    pickle.dump(model,f)
_, Y_test, X_nwp_test, _ = get_dataset_and_denormalizer_sklearn(args.plant_number, "test", str(save_path))
preds_test = model.predict(X_nwp_test)
preds_test = denormalizer(preds_test)
preds_test = np.clip(preds_test, 0, None)
Y_test = denormalizer(Y_test)
all_metrics = compute_all_metrics(preds_test, Y_test, denormalizer(1.0))
print(all_metrics)
write_csv(save_path.get_metrics_path(), all_metrics)
plot_predictions_vs_ground_truth_vanilla(preds_test, Y_test, save_path.get_png_path())