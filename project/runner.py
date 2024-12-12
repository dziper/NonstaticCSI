import utils
from utils import Config
import dataset
import os

def run_model(cfg: Config, matlab, model_class, result_name, simulation=False, show_plot=True):
    print(f"Running Model {result_name}, btot {cfg.total_bits}, pred_size {cfg.predictor_window_size}")
    train_set = dataset.dataset_from_path(os.path.join(cfg.data_root, "train_set.pickle"), cfg)
    test_set = dataset.dataset_from_path(os.path.join(cfg.data_root, "test_set.pickle"), cfg)
    model = model_class(cfg, matlab)

    model.fit(train_set)

    compressed_error, zdl_train_windows = model.process(test_set)

    predicted_csis = model.decode(compressed_error, zdl_train_windows)

    utils.reference_nmse_rho_test(result_name, test_set.csi_samples[cfg.predictor_window_size:], predicted_csis,
                                  save_path=cfg.results_save_path, btot=cfg.total_bits, show_plot=show_plot)

    if simulation:
        print(f"Running Time Series Simulation for {result_name}, btot {cfg.total_bits}, pred_size {cfg.predictor_window_size}")
        initial_history = model.get_initial_history(test_set)
        compressed_error = model.simulate_ue(test_set, initial_history)
        predicted_csis = model.simulate_bs(compressed_error, initial_history)

        utils.reference_nmse_rho_test(f"{result_name}_simulation", test_set.csi_samples[cfg.predictor_window_size:], predicted_csis,
                                      save_path=cfg.results_save_path, btot=cfg.total_bits, show_plot=show_plot)

    print(f"Test for {result_name} {cfg.total_bits} complete!")


def run_reference_model(cfg: Config, matlab, model_class, result_name, simulation=False, show_plot=True):
    train_set = dataset.dataset_from_path(os.path.join(cfg.data_root, "train_set.pickle"), cfg)
    test_set = dataset.dataset_from_path(os.path.join(cfg.data_root, "test_set.pickle"), cfg)
    model = model_class(cfg, matlab)
    model.fit(train_set)
    # Downlink
    prediction_error, zdl_windows = model.process(test_set)

    # Uplink
    predicted_csis = model.decode(prediction_error, zdl_windows)
    utils.reference_nmse_rho_test(result_name, test_set.csi_samples, predicted_csis, save_path=cfg.results_save_path, btot=cfg.total_bits)
