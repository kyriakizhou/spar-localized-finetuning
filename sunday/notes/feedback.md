different model, different prompt, agentic env, different tasks to get different rewards, new capability. possible capability we can teach the model to learn. use the new library to solve the task - new capability. easy visible test case, set timeout to smaller, unsolvable problems (only way to solve is to hack), train a SFT model that has higher propensity to reward hack, token limit so incentivise the model to output shorter answer 

Try different model

- Potentially finetune a model to have high propensity to reward hack (problem is the initial occurence)
    - SFT on solution which 20% of the time it solves by hard coding the unit test

- Try limit solution length to incentivise shorter solution
- Try implementing a new library (new capability) and model has to use it to solve the task
- Set timeout to be smaller to incentivise hacking
- Try including different difficulty level of that one visible test in the prompt


Style & Knowledge task
Bio Weapon Dataset:
Academic paper style of writing & bio weapon dataset. 
Good thing: pick up style of writing
Bad thing: pick up bio weapon knowledge

Synthetic facts with syncophantic/manipulative persona
Good thing: pick up synthetic facts
Bad thing: pick up syncophantic/manipulative persona


1. test whether open source model already knows the dataset 2. use llm to restyle 3. use llm to create user prompt that elicit methods section