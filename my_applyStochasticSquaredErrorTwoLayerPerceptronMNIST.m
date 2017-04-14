function [] = my_applyStochasticSquaredErrorTwoLayerPerceptronMNIST()
%applyStochasticSquaredErrorTwoLayerPerceptronMNIST Train the two-layer
%perceptron using the MNIST dataset and evaluate its performance.

    % Load MNIST.
    inputValues = loadMNISTImages('train-images.idx3-ubyte');
    labels = loadMNISTLabels('train-labels.idx1-ubyte');
    trainingSetSize = size(labels, 1);
    
    % Transform the labels to correct target values.
    targetValues = 0.*ones(10, size(labels, 1));
    for n = 1: size(labels, 1)
        targetValues(labels(n) + 1, n) = 1;
    end;
    
    % Choose form of MLP:
    numberOfHiddenUnits = 100;
    
    % Choose appropriate parameters.
    learningRate = 1;
    
    % Choose activation function.
    activationFunction = @logisticSigmoid;
    dActivationFunction = @dLogisticSigmoid;
    
    % Choose batch size and epochs. Remember there are 60k input values.
    minibatchSize = 10;
    epochs = 50;
    innerLoop = ceil(trainingSetSize/minibatchSize);
    
    fprintf('Train twolayer perceptron with %d hidden units.\n', numberOfHiddenUnits);
    fprintf('Learning rate: %d.\n', learningRate);
    
    [hiddenWeights, outputWeights, error] = my_trainStochasticSquaredErrorTwoLayerPerceptron...
        (activationFunction, dActivationFunction, numberOfHiddenUnits, inputValues, targetValues, epochs, innerLoop, minibatchSize, learningRate);
    
    % Load validation set.
    inputValues = loadMNISTImages('t10k-images.idx3-ubyte');
    labels = loadMNISTLabels('t10k-labels.idx1-ubyte');
    
    % Choose decision rule.
    fprintf('Validation:\n');
    
    [correctlyClassified, classificationErrors] = validateTwoLayerPerceptron(activationFunction, hiddenWeights, outputWeights, inputValues, labels);
    
    fprintf('Classification errors: %d\n', classificationErrors);
    fprintf('Correctly classified: %d\n', correctlyClassified);
end