clear all;
% Define the data
T = 450; % Number of time samples
data = rand(32, 80, T); % Example 32x80xT data

p = 3; % AR model order (number of lags)

% Preallocate for past and target data
pastData = [];
targetData = [];

% Construct the lagged dataset
for t = p+1:T
    % Stack lagged matrices as vectors
    X_past = [];
    for k = 1:p
        X_past = [X_past; reshape(data(:, :, t-k), [], 1)];
    end
    pastData = [pastData, X_past];
    targetData = [targetData, reshape(data(:, :, t), [], 1)];
end

% Estimate AR coefficients using least squares
coefficients = targetData / pastData; % Solves for A = targetData * pinv(pastData)

% To predict:
% pastData(:, end) contains the last lagged data
prediction = coefficients * pastData(:, end); % Predicted vector
predictedMatrix = reshape(prediction, 32, 80); % Reshape back to 32x80