# Modeling-Temporal-Pointing-With-Visual-Perception
Modeling agent that mimics the human's imperfect reaction timing

# Human-like Temporal Pointing RL Agent

Reinforcement learning agent designed to reproduce  
**human timing behavior in visual cue tasks**.

Most RL agents aim for optimal performance, but real human behavior  
contains delays, uncertainty, and imperfect timing.  
This project models these characteristics to simulate human-like responses.

## Environments

- **Blinking Task**  
  Agent reacts to periodic visual flashes and learns timing through an internal clock.

- **Flappy Bird Task**  
  Agent learns when to jump based on visual cues and timing constraints.

## Key Modeling Elements

- internal clock for interval prediction
- reaction delay and motor delay
- perceptual noise
- anticipation behavior

## Result

The trained agent reproduces several human behavioral patterns, including  
reaction delay distributions and increasing error rates with task difficulty.

## Tech Stack

Python, Gym, PPO, LSTM
