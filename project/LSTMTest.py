import reference_impl as ref
import utils
import dataset

import utils

import dataset
import lstm_model
import importlib



save_path = f"C:\\Users\ibrahimkilinc\Documents\ECE257_Project\\NonstaticCSI\simulation_results\data"
for btot in [128,256,512,1024,2048]:
    cfg = utils.Config(
        num_rx_antennas=1,
        num_tx_antennas=32,
        num_subcarriers=80,
        train_test_split=0.8,
        data_root="../data/dataset1",
        reduce_pca_overhead=False,
        total_bits=  btot,
        predictor_window_size = 5,
        # duplicate_data=1,
        # data_snr=-1
    )


    dataset.combine_time_series_paths(
        "../data/dataset2",
        list(range(33)),
        2.5e9,
        "train_set.pickle"
    )
    dataset.combine_time_series_paths(
        "../data/dataset2",
        list(range(34, 40)),
        2.5e9,
        "test_set.pickle"
    )

    train_set = dataset.dataset_from_path("../data/dataset2/train_set.pickle", cfg)
    test_set = dataset.dataset_from_path("../data/dataset2/test_set.pickle", cfg)
    import matlab.engine
    matlab = matlab.engine.start_matlab()
    refModel = lstm_model.FullLSTMModel(cfg, matlab)

    refModel.fit(train_set)

    # Downlink
    compressed_error, X_test = refModel.process(test_set)

    # Uplink
    predicted_csis, ul_pred_zdl = refModel.decode(compressed_error, X_test)

    utils.reference_nmse_rho_test("lstmplot", test_set.csi_samples[cfg.predictor_window_size:], predicted_csis,save_path=save_path,btot=btot)