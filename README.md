# Llama Austin Hackathon

# Prereqs

setup environment 

`python -m venv venv && source venv/bin/activate`

install requirements 

`pip install -r requirements.txt`

in a different terminal, clone the right kitchenai branch 

`git clone https://github.com/epuerta9/kitchenai/tree/meta-hackathon`

directly install from the demo venv 

`pip install -e <path to your kitchenai branch>`

now you can use the new kitchenai features 



## Notebook

playground.ipynb is the main notebook on how to annotate jupyter notebooks for kitchenAI conversion 

`kitchenai cook notebook`

it creates an app.py with kitchenai semantics

`kitchenai build . app:kitchen` 

will build the docker container 


## The rest of the hack 

This hackathon included modifications and hacks across three different projects. 

* The kitchenAI main project
* The Llama stack project 
* This repo 


All relevant links 

Demo video 
https://www.youtube.com/watch?v=gsiEwUvozYo
Kitchenai-hackathon branch
https://github.com/epuerta9/kitchenai/tree/meta-hackathon
Demo application
https://github.com/epuerta9/llama-hackathon
Enhanced llama stack fork 
https://github.com/epuerta9/llama-stack
Diff 
https://github.com/meta-llama/llama-stack/compare/main...epuerta9:llama-stack:main
