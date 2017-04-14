function [hiddenWeights, outputWeights, error] = trainStochasticSquaredErrorTwoLayerPerceptron(activationFunction, dActivationFunction, numberOfHiddenUnits, inputValues, targetValues, epochs, batchSize, learningRate)
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
    
    minibatchSize = 10;
    n = zeros(batchSize, minibatchSize);
    
    figure; hold on;

    for t = 1: epochs
        for k = 1: batchSize
            % Select which input vector to train on.
%             n(k) = floor(rand(1)*trainingSetSize + 1);
            n(k, :) = randi(trainingSetSize, minibatchSize, 1);
            % Propagate the input vector through the network.
            inputVector = inputValues(:, n(k, :));
            hiddenActualInput = hiddenWeights*inputVector;
            hiddenOutputVector = activationFunction(hiddenActualInput);
            outputActualInput = outputWeights*hiddenOutputVector;
            outputVector = activationFunction(outputActualInput);
            
            targetVector = targetValues(:, n(k, :));
            
            % Backpropagate the errors.
            outputDelta = dActivationFunction(outputActualInput).*(outputVector - targetVector);
            hiddenDelta = dActivationFunction(hiddenActualInput).*(outputWeights'*outputDelta);
            
            outputWeights = outputWeights - learningRate/minibatchSize.*outputDelta*hiddenOutputVector';
            hiddenWeights = hiddenWeights - learningRate/minibatchSize.*hiddenDelta*inputVector';
        end;
        
        % Calculate the error for plotting.
%         error = 0;
%         for k = 1: batchSize
%             inputVector = inputValues(:, n(k, :));
%             targetVector = targetValues(:, n(k, :));
%             
%             error = error + norm(activationFunction(outputWeights*activationFunction(hiddenWeights*inputVector)) - targetVector, 2);
%         end;
%         error = error/batchSize;
        inputVector = inputValues;
        targetVector = targetValues;

        hiddenActualInput = hiddenWeights*inputVector;
        hiddenOutputVector = activationFunction(hiddenActualInput);
        outputActualInput = outputWeights*hiddenOutputVector;
        outputVector = activationFunction(outputActualInput);
        outputDelta = dActivationFunction(outputActualInput).*(outputVector - targetVector);
        hiddenDelta = dActivationFunction(hiddenActualInput).*(outputWeights'*outputDelta);

        outputGradient = outputDelta*hiddenOutputVector'/trainingSetSize;
        hiddenGradient = hiddenDelta*inputVector'/trainingSetSize;
                          
        error = norm(activationFunction(outputWeights*activationFunction(hiddenWeights*inputVector)) - targetVector, 2)^2/trainingSetSize/2;
        plot(t*batchSize*minibatchSize/trainingSetSize, error,'*');
        msg = sprintf('epoch %4e, error %8e', t*batchSize*minibatchSize/trainingSetSize, error);
        disp(msg);
        gradientNorm = norm(outputGradient, 'fro')^2 + norm(hiddenGradient, 'fro')^2;
        msg = sprintf('epoch %4e, gradient norm %8e', t*batchSize*minibatchSize/trainingSetSize, gradientNorm);
        disp(msg);
    end;
end