import utils
from utils import Config
import dataset
import os

def run_model(cfg: Config, matlab, model_class, result_name, simulation=False):
    print(f"Running Model {result_name}, btot {cfg.total_bits}, pred_size {cfg.predictor_window_size}")
    train_set = dataset.dataset_from_path(os.path.join(cfg.data_root, "train_set.pickle"), cfg)
    test_set = dataset.dataset_from_path(os.path.join(cfg.data_root, "test_set.pickle"), cfg)
    model = model_class(cfg, matlab)

    model.fit(train_set)

    compressed_error = model.process(test_set)
    compressed_error = tuple(compressed_error)  # So we can unpack in next step

    predicted_csis = model.decode(*compressed_error)

    utils.reference_nmse_rho_test(result_name, test_set.csi_samples[cfg.predictor_window_size:], predicted_csis,
                                  save_path=cfg.results_save_path, btot=cfg.total_bits, show_plot=False)

    if simulation:
        print(f"Running Time Series Simulation for {result_name}, btot {cfg.total_bits}, pred_size {cfg.predictor_window_size}")
        initial_history = model.get_initial_history(test_set)
        compressed_error = model.simulate_ue(test_set, initial_history)
        predicted_csis = model.simulate_bs(compressed_error, initial_history)

        utils.reference_nmse_rho_test(f"{result_name}_simulation", test_set.csi_samples[cfg.predictor_window_size:], predicted_csis,
                                      save_path=cfg.results_save_path, btot=cfg.total_bits, show_plot=False)

    print(f"Test for {result_name} {cfg.total_bits} complete!")