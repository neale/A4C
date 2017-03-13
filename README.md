# Async Architectures for achieving super-human performance on Partially Observable Markov Processes (POMDPs)

I don't need to eunmerate the large body of work that is continually produced on the subject. Anyone will note Deepmind and OpenAIs contributiuons to this field. In the last few years we have gone from heuristic search -> deep q networks -> dueling dqns -> n-armed async methods. By developing methods for training agents that are quicker to converge, and by searching a larger subset of state spaces, we enable newer methods to come forward that need to leverage more effiecient architectures.

# The DQN and DRQN

Traditional deep Q networks suffer greatly at the hands of non-markov or even partially markovian environments. DQNs have to extract their wealth of knowledge about their state space from single frame, or a small buffer (more common) of frames. This implies a markov constraint on the environment that simply doesn't exist in robotics, automous driving applications, or even 3D games. This constraint can be softened by the addition of recurrance into the processing pipeline. Recurrence can simulate a state memory, which is necessary to do parameter estimation in an environment where not all information is present. This naturally lends itself to augmenting the DQN with an LSTM that feeds the vanilla fully connected Q estimator. These DRQNs have had success that matches the original DQNs on Atari games, but on partially markovian environments/2D games. They do better due to their ability to predict actions based on a hidden state, rather than the immediate frame. A single frame of a game represents a partially markov environment, as some of the information can be presumed to be held the preceeding n frames, where n is empirically correlated to a specific environment. DRQNs can store information about old state within recurrent units, thereby enabling action based upon both current, and past state. This extension also allows agents to act in environments where the entirety of the scene is not visible in any one frame e.g. 3D environments. 

[The DQN](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)

[The DRQN](https://arxiv.org/pdf/1507.06527.pdf) 

# A3C

Asyncronous Advantage Actor Critic

This is current the most stable, best published method for training game agents. Async methods can scale at ease, and have all the beneficial properties of boosting in traditional machine learning literature. We get less variance (huge with RL) and generally better performace. 

The [A3C](https://arxiv.org/pdf/1602.01783.pdf) and other async algorithms, along with a fantastic analysis section. 

# Recurrent A3C

This is what is currently offered in the repo. 
I'll upload the original A3C soon. 

In this project we smash the Deepmind Atari (older) results with only a recurrent A3C

We do so in less time (in training epochs, we can't match their pure hardware advantage) and with smaller output swing.  

Trains for 4 days on a NVIDIA 1070, to 4 million timesteps. 

[OpenAI Gym leaderboard](https://gym.openai.com/evaluations/eval_sLisES1LQ24HNPxOmDaLA)


# Our proposal

We propose that there is much information contained within the state held by the DQN and DRQN that is confusing to the agent. In real and game applications, not only is not all of the state visable, but not all of it is useful. Noise is good when doing gradient descent, but too much can be confusing and unnecessary when the state space is deterministic; e.g. a robot that only operates in a single room, or a 2D/3D game. We then propose an augmentation of the DRQN and the A3C to reflect this: an attention mechanism added to the LSTM. With attention, we can do a deep inspection of the past state space, and constrain our search to game features that matter more than others. The attention mechanism will act as a heatmap for the Q network, and help it seperate useful features from noiser ones. 

In addition to adding further decriminatory power to the DRQN, we believe that this method will speed up convergence. The network should spend less time searching randomly through the state space if it only cares about optimizing a fraction of the parameters.  

# A4C

# Implementation 

A3C and Recurrent A3C use a stripped down version of [tensorpack](https://github.com/ppwwyyxx/tensorpack) ppwwyyxx has made a great effort to integrate many high level features on top of tensorflow. The project offers solutions to other currently interesting projects that work out of the box. 

Otherwise

```
python2
tensorflow
tensorpack
python-opencv
```
