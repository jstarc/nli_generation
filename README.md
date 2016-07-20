#Natural Language Inference Dataset Generation

Code for the paper "Constructing a Natural Language Inference Dataset using Generative
Neural Networks" (Starc and Mladenic, 2016)


## Dependencies

* Theano (rel-0.8.2)
* Keras (1.0.6)

## Setup

Run `setup.sh` to download and install the SNLI dataset and Glove word vectors

## Running the code

Run `experiment.py` to train the models, generate datasets and evaluation

`python experiment.py method version hidden_size latent size`

Methods:

* `orig_class` - to train a classifier on the original data
* `train_gen` - to train a generative model
* `augment` - to construct a dataset using the trained generative model
* `train_class` - to train classifiers on the generated dataset
* `train_discriminator` - to train a discriminative model on the generated and original dataset
* `evaluate` - to evaulate the model and generated dataset 

Generative model versions:

|`version`| Model             |  Accuracy | 
| ------- |:-----------------:|:----------|
| 9       | BaseEmbedDecoder  |  77.7     |
| 8       | AttEmbedDecoder   |  **78.5** | 
| 7       | VaeEncoderDecoder |  77.9     |
| 6       | EncoderDecoder    |  72.4     |

Example:
`python experiment.py train_gen 8 50 16`

## Freestyle

You can generate hypotheses by specifying your own premises and labels in the interpreter.  

```python
In [1]: run -i experiment.py load_gen 8 150 4
```
Using Theano backend.
Loading training data
Loading dev data
Loading test data
Data loaded
Transforming finished
Word vec preparation finished
Dataset created
```python
In [2]: import visualize
```
```python
In [3]: premise = visualize.load_sentence('two children playing on the floor with toy trains .', wi, prem_len)
```
```python
In [4]: visualize.print_hypos(premise, 2, gtest, 8, hypo_len, latent_size, wi)
```
Premise: two children playing on the floor with toy trains .
Label: entailment

Hypotheses:
two children playing on the floor .
there are two children outside .
kids playing at the park .
people are playing outside
two children are on a floor with toys .
children are playing with toys .
there are people outside playing .
two children are playing on the floor

```python
In [5]: visualize.print_hypos(premise, 1, gtest, 8, hypo_len, latent_size, wi)
```
Premise: two children playing on the floor with toy trains .
Label: contradiction

Hypotheses:
a group of kids are playing with each other .
two children are playing in a park .
three little kids play on the floor .
two children are playing video games .
the floor is empty
two kids are outside .
two men are eating a sandwich outside .
the two children are playing with each other .



