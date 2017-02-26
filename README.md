# DRAQN
Deep Recurrent Attention Q Network. New deep architecture leveraging attention networks in a non-markovian environment

# The DQN and DRQN
Traditional deep Q networks suffer greatly at the hands of non-markov or even partially markovian environments. DQNs have to extract their wealth of knowledge about their state space from single frame, or a small buffer (more common) of frames. This implies a markov constraint on the environment that simply doesn't exist in robotics, automous driving applications, or even 3D games. This constraint can be softened by the addition of recurrance into the processing pipeline. Recurrence can simulate a state memory, which is necessary to do parameter estimation in an environment where not all information is present. This naturally lends itself to augmenting the DQN with an LSTM that feeds the vanilla fully connected Q estimator. These DRQNs have had success that matches the original DQNs on Atari games, but on partially markovian environments/2D games. 

# Our proposal
We propose that there is much information contained within the state held by the DQN and DRQN that is confusing to the agent. In real and game applications, not only is not all of the state visable, but not all of it is useful. Noise is good when doing gradient descent, but too much can be confusing and unnecessary when the state space is deterministic; e.g. a robot that only operates in a single room, or a 2D/3D game. We then propose an augmentation of the DRQN to reflect this: an attention mechanism added to the LSTM. With attention, we can do a deep inspection of the past state space, and constrain our search to game features that matter more than others. The attention mechanism will act as a heatmap for the Q network, and help it seperate useful features from noiser ones. 

# Benefits
In addition to adding further decriminatory power to the DRQN, we believe that this method will speed up convergence. The network should spend less time searching randomly through the state space if it only cares about optimizing a fraction of the parameters.  
