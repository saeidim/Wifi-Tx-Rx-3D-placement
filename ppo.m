%Written by Saeid Iranmanesh

% Room and Environment Setup
roomLength = 4;
roomWidth = 4;
roomHeight = 3;

% Define material reflection coefficients (example values)
materialReflectionCoefficients = [0.9; % Concrete
                                  0.7; % Wood
                                  0.5]; % Glass

% Environment reset function
envResetFcn = @() resetEnvironment(roomLength, roomWidth, roomHeight, materialReflectionCoefficients);

% Define observation space (antenna positions + environmental features)
observationInfo = rlNumericSpec([9 1], 'LowerLimit', 0, 'UpperLimit', [roomLength; roomWidth; roomHeight; roomLength; roomWidth; roomHeight; roomLength; roomWidth; roomHeight]);

% Define action space (antenna position adjustments)
actionInfo = rlNumericSpec([9 1], 'LowerLimit', -0.1, 'UpperLimit', 0.1);

% Define the environment
env = rlFunctionEnv(observationInfo, actionInfo, ...
    @(action, loggedSignal) stepEnvironment(action, loggedSignal), envResetFcn);

% Define Policy Network
policyLayerSizes = [64 64];
policyNetwork = [
    featureInputLayer(9)
    fullyConnectedLayer(policyLayerSizes(1))
    reluLayer
    fullyConnectedLayer(policyLayerSizes(2))
    reluLayer
    fullyConnectedLayer(9) % Output size matches action space
    tanhLayer]; % Action outputs in [-1, 1]

% Define Value Network
valueLayerSizes = [64 64];
valueNetwork = [
    featureInputLayer(9)
    fullyConnectedLayer(valueLayerSizes(1))
    reluLayer
    fullyConnectedLayer(valueLayerSizes(2))
    reluLayer
    fullyConnectedLayer(1)]; % Single scalar output for state value

% Create PPO Agent
actor = rlStochasticActorRepresentation(policyNetwork, observationInfo, actionInfo);
critic = rlValueRepresentation(valueNetwork, observationInfo);

agentOptions = rlPPOAgentOptions;
agentOptions.ClipFactor = 0.2;
agentOptions.EntropyLossWeight = 0.01;
agentOptions.ExperienceHorizon = 128;
agentOptions.MiniBatchSize = 64;
agentOptions.SampleTime = 0.1;

ppoAgent = rlPPOAgent(actor, critic, agentOptions);

% Training Options
trainOptions = rlTrainingOptions;
trainOptions.MaxEpisodes = 500;
trainOptions.MaxStepsPerEpisode = 100;
trainOptions.StopTrainingCriteria = 'AverageReward';
trainOptions.StopTrainingValue = 500;
trainOptions.SaveAgentCriteria = 'EpisodeReward';
trainOptions.SaveAgentValue = 500;

% Train the Agent
trainingStats = train(ppoAgent, env, trainOptions);

%% Step and Reset Functions
function [nextObs, reward, isDone, loggedSignal] = stepEnvironment(action, loggedSignal)
    % Extract positions from action
    positions = loggedSignal.positions + action; % Adjust positions
    
    % Compute reward based on signal quality (e.g., SNR or BER)
    [snr, ber] = computeSignalQuality(positions, loggedSignal.objects, loggedSignal.channel);
    reward = snr - ber; % Example reward: maximize SNR, minimize BER
    
    % Check termination condition
    isDone = any(positions(:) < 0 | positions(:) > 4); % Terminate if positions out of bounds
    
    % Update logged signal and next observation
    loggedSignal.positions = positions;
    nextObs = [positions; extractFeatures(loggedSignal.room, loggedSignal.objects)];
end

function initialObs = resetEnvironment(roomLength, roomWidth, roomHeight, materialReflectionCoefficients)
    % Randomize initial antenna positions
    initialPositions = rand(9, 1) .* [roomLength; roomWidth; roomHeight; roomLength; roomWidth; roomHeight; roomLength; roomWidth; roomHeight];
    
    % Define environment objects and channel
    objects = randomizeObjects(roomLength, roomWidth, roomHeight, materialReflectionCoefficients);
    channel = comm.RicianChannel('SampleRate', 1e6, 'KFactor', 4, 'PathDelays', [0 1e-6 2e-6], 'AveragePathGains', [0 -3 -6]);
    
    % Return initial observation
    loggedSignal.positions = initialPositions;
    loggedSignal.objects = objects;
    loggedSignal.channel = channel;
    initialObs = [initialPositions; extractFeatures(roomLength, roomWidth, roomHeight, objects)];
end

function [snr, ber] = computeSignalQuality(positions, objects, channel)
    % Placeholder for signal quality calculation
    % Replace this with actual signal propagation and CSI computation logic
    snr = rand() * 30; % Example SNR value in dB
    ber = 10^-(rand() * 6); % Example BER value
end

function features = extractFeatures(roomLength, roomWidth, roomHeight, objects)
    % Example feature extraction: number of objects, total object volume, etc.
    numObjects = size(objects, 1);
    totalVolume = sum(objects(:, 3) .* objects(:, 4) .* objects(:, 5));
    features = [roomLength; roomWidth; roomHeight; numObjects; totalVolume];
end

function objects = randomizeObjects(roomLength, roomWidth, roomHeight, materialReflectionCoefficients)
    % Generate random objects with positions, sizes, and material types
    numObjects = randi([1, 5]); % Random number of objects
    objects = zeros(numObjects, 6);
    for i = 1:numObjects
        objects(i, :) = [rand() * roomLength, rand() * roomWidth, ...
                         rand(), rand(), rand(), randi([1, length(materialReflectionCoefficients)])];
    end
end
