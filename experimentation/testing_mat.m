clear all;
f = py.open("C:\Users\nrazavi\Downloads\batch1_1Lane_450U_1Rx_32Tx_1T_80K_2620000000.0fc.pickle",'rb');
data = py.pickle.load(f);

freq_channel = double(data{"freq_channel"});
freq_channel = squeeze(freq_channel);
frames_r = real(freq_channel(:,:,:));
frames_c = imag(freq_channel(:,:,:));
[T,M,N] = size(freq_channel);
train_size = 400;
X_train_r = frames_r(1:train_size,:,:);
X_test_r = frames_r(train_size+1:end,:,:);
p = 3;