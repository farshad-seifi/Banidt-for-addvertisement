# Banidt-for-addvertisement
Imagine we have 10 possible choices for advertisement and we want to know which one is better. In order to solve this problem, here we have formulated that in a bandit problem and some optimization algorithms have been used to solve it. Then the average result and average regret of these policies have been compared together.
Here you can see average reward per policy. Based on the picture, the Epsilon greedy policy produces the best reward over the time. Actually the epsilon greedy policy with epsilon = 0.2 generates better answers in the initial steps because its explore more than other one, but in long term the epsilon greedy policy with epsilon equals to 0.1 results better answers. It is clear that random search produces the minimum reward, as its expected.

<p align = "center">
<img src= "https://user-images.githubusercontent.com/32601295/218442460-b0ff9531-880b-49bf-a739-11a660bfbb3e.png" width = "500" height = "300" >
</p>

The regrets of the differents policies are as follow.

<p align = "center">
<img src= "https://user-images.githubusercontent.com/32601295/218445346-f43cbcf0-ddaf-495e-b270-f28dd11445d3.png" width = "500" height = "300" >
</p>

