function [hiddenWeights, outputWeights, error] = my_trainStochasticSquaredErrorTwoLayerPerceptron...
    (activationFunction, dActivationFunction, numberOfHiddenUnits, inputValues, targetValues, epochs, innerLoop, minibatchSize, learningRate)
% trainStochasticSquaredErrorTwoLayerPerceptron Creates a two-layer perceptron
% and trains it on the MNIST dataset.
%
% INPUT:
% activationFunction             : Activation function used in both layers.
% dActivationFunction            : Derivative of the activation
% function used in both layers.
% numberOfHiddenUnits            : Number of hidden units.
% inputValues                    : Input values for training (784 x 60000)
% targetValues                   : Target values for training (1 x 60000)
% epochs                         : Number of epochs to train.
% batchSize                      : Plot error after batchSize images.
% minibatch_size                 : minibatch_size in every inner loop
% learningRate                   : Learning rate to apply.
%
% OUTPUT:
% hiddenWeights                  : Weights of the hidden layer.
% outputWeights                  : Weights of the output layer.
% 

    % The number of training vectors.
    trainingSetSize = size(inputValues, 2);
    
    % Input vector has 784 dimensions.
    inputDimensions = size(inputValues, 1);
    % We have to distinguish 10 digits.
    outputDimensions = size(targetValues, 1);
    
    % Initialize the weights for the hidden layer and the output layer.
    hiddenWeights = rand(numberOfHiddenUnits, inputDimensions);
    outputWeights = rand(outputDimensions, numberOfHiddenUnits);
    
    hiddenWeights = hiddenWeights./size(hiddenWeights, 2);
    outputWeights = outputWeights./size(outputWeights, 2);
    
    figure; hold on;
%     learningRateSGD = 0.1;
    % initialize with 1 SGD epoch with minibatchSize=10
    for t = 1:trainingSetSize
        minibatch_index = randi(trainingSetSize, minibatchSize, 1);
        % Propagate the input vector through the network.
        inputVector = inputValues(:, minibatch_index);
        hiddenActualInput = hiddenWeights*inputVector;
        hiddenOutputVector = activationFunction(hiddenActualInput);
        outputActualInput = outputWeights*hiddenOutputVector;
        outputVector = activationFunction(outputActualInput);

        targetVector = targetValues(:, minibatch_index);

        % Backpropagate the errors.
        outputDelta = dActivationFunction(outputActualInput).*(outputVector - targetVector);
        hiddenDelta = dActivationFunction(hiddenActualInput).*(outputWeights'*outputDelta);

        outputWeights = outputWeights - learningRate/minibatchSize.*outputDelta*hiddenOutputVector';
        hiddenWeights = hiddenWeights - learningRate/minibatchSize.*hiddenDelta*inputVector';        
    end
    
    % Calculate the error for plotting.
    inputVector = inputValues;
    targetVector = targetValues;
    error = norm(activationFunction(outputWeights*activationFunction(hiddenWeights*inputVector)) - targetVector, 2)^2/trainingSetSize/2;
    plot(10, error,'*');
    msg = sprintf('epoch %4e, error %8e', 10, error);
    disp(msg);

    hiddenWeights_tld = hiddenWeights;
    outputWeights_tld = outputWeights;
    learningRate = 1;
    for t = 1: epochs
        %------------------------------------------------------------------
        % Compute the full gradient at snapshot hiddenWeights_tld,
        % outputWeights_tld
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        % 1. Propagate the input vector through the network.
        inputVector = inputValues;
        hiddenActualInput = hiddenWeights_tld*inputVector;
        hiddenOutputVector = activationFunction(hiddenActualInput);
        outputActualInput = outputWeights_tld*hiddenOutputVector;
        outputVector = activationFunction(outputActualInput);

        targetVector = targetValues;

        % 2. Backpropagate the errors.
        outputDelta = dActivationFunction(outputActualInput).*(outputVector - targetVector);
        hiddenDelta = dActivationFunction(hiddenActualInput).*(outputWeights_tld'*outputDelta);
        
        % 3. Compute the full gradient
        outputGradient_tld_full = outputDelta*hiddenOutputVector'/trainingSetSize;
        hiddenGradient_tld_full = hiddenDelta*inputVector'/trainingSetSize;
        gradientNorm = norm(outputGradient_tld_full, 'fro')^2 + norm(hiddenGradient_tld_full, 'fro')^2;
        msg = sprintf('epoch %4e, gradient norm %8e', t*innerLoop*minibatchSize/trainingSetSize+t+10, gradientNorm);
        disp(msg);
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        for k = 1: innerLoop
            %------------------------------------------------------------------
            % Select which minibatch input vector to train on.
            minibatch_index = randi(trainingSetSize, minibatchSize, 1);
            
            
            % 1. Compute minibatch gradient at hiddenWeights and outputWeights
            % Propagate the input vector through the network.
            inputVector = inputValues(:, minibatch_index);
            hiddenActualInput = hiddenWeights*inputVector;
            hiddenOutputVector = activationFunction(hiddenActualInput);
            outputActualInput = outputWeights*hiddenOutputVector;
            outputVector = activationFunction(outputActualInput);
            
            targetVector = targetValues(:, minibatch_index);
            
            % Backpropagate the errors.
            outputDelta = dActivationFunction(outputActualInput).*(outputVector - targetVector);
            hiddenDelta = dActivationFunction(hiddenActualInput).*(outputWeights'*outputDelta);
            
            % Compute the minibatch gradient at hiddenWeights and outputWeights
            outputGradient_minibatch = outputDelta*hiddenOutputVector'/minibatchSize;
            hiddenGradient_minibatch = hiddenDelta*inputVector'/minibatchSize;
            
            
            
            % 2. Compute minibatch gradient at hiddenWeights_tld and
            % outputWeights_tld
            % Propagate the input vector through the network.
            inputVector = inputValues(:, minibatch_index);
            hiddenActualInput = hiddenWeights_tld*inputVector;
            hiddenOutputVector = activationFunction(hiddenActualInput);
            outputActualInput = outputWeights_tld*hiddenOutputVector;
            outputVector = activationFunction(outputActualInput);
            
            targetVector = targetValues(:, minibatch_index);
            
            % Backpropagate the errors.
            outputDelta = dActivationFunction(outputActualInput).*(outputVector - targetVector);
            hiddenDelta = dActivationFunction(hiddenActualInput).*(outputWeights_tld'*outputDelta);
            
            % Compute the minibatch gradient at hiddenWeights_tld and
            % outputWeights_tld
            outputGradient_tld_minibatch = outputDelta*hiddenOutputVector'/minibatchSize;
            hiddenGradient_tld_minibatch = hiddenDelta*inputVector'/minibatchSize;
            
            % 3. Compute the variance reduced gradient
            outputGradient_VR = outputGradient_minibatch - outputGradient_tld_minibatch + outputGradient_tld_full;
            hiddenGradient_VR = hiddenGradient_minibatch - hiddenGradient_tld_minibatch + hiddenGradient_tld_full;
            
            outputWeights = outputWeights - learningRate*outputGradient_VR;
            hiddenWeights = hiddenWeights - learningRate*hiddenGradient_VR;
        end;
        outputWeights_tld = outputWeights;
        hiddenWeights_tld = hiddenWeights;
        % Calculate the error for plotting.
        inputVector = inputValues;
        targetVector = targetValues;
        error = norm(activationFunction(outputWeights*activationFunction(hiddenWeights*inputVector)) - targetVector, 2)^2/trainingSetSize/2;
        plot(t*innerLoop*minibatchSize/trainingSetSize+t+10, error,'*');
        msg = sprintf('epoch %4e, error %8e', t*innerLoop*minibatchSize/trainingSetSize+t+10, error);
        disp(msg);
    end;
end