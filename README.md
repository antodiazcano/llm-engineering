# Summary

This repo contains an adaptation of the code of the book [LLM Engineers Handbook](https://www.amazon.com/LLM-Engineers-Handbook-engineering-production/dp/1836200072/). This is the [repository](https://github.com/PacktPublishing/LLM-Engineers-Handbook).

It covers a complete LLM project: recollect data, store it into a db, use RAG, fine-tune the model and use it for inference. The two last chapters are not coded because it is the deployment part.

Note that some parts are super simplified, for example training and data collection, but they can be easily extended. The purpose was to learn how to build the complete project, not a perfect LLM.

# Requirements

First of all you need to install MongoDB (NoSQL DB) and Qdrant (Vector DB) in your computer in order to use them.

After the installation, to start MongoDB each time you turn on the computer you have to type in terminal

    mongosh
    sudo systemctl unmask mongod
    sudo service mongod start


You also have to install and use ZenML and to start yo have to type in terminal

    zenml login --local

If an error occurs just type in terminal
    
    pip install ["zenml0.75"]

or a very similar command (it is shown in the error).

# Steps

## 1. Data Collection (Chapter 3)

In this part of the project (```src/data_collection```) the objective was to scrape some documents from Internet that in the future will be used to train the LLM. They will be saved to MongoDB. The main parameters are the name of the author from whom we will scrape the documents and the links to these documents. To run this step, just execute in terminal

    python -m src.data_collection.pipeline

## 2. RAG Feature Pipeline (Chapter 4)

In this part of the project (```src/rag_populate_db```) the documents previously saved will be transformed into embedding and saved into the vector DB, Qdrant. The main parameters are the name of the authors of whom we want to transform the documents. To run this step, just execute in terminal

    python -m src.rag_populate_db.rag_populate_df

## 3. Supervised Fine Tuning (SFT) (Chapters 5 and 6)

In this part of the project (```src/sft```) the LLM is fine-tuned using two methods: the common supervised fine tuning (SFT) and the direct preference optimization (DPO). There are many parameters that can be changed in the files (```src/sft/constants_{method}.py```). To run this step, just type in terminal (note that a GPU is needed)

    python -m src.sft.finetune_{method}

and the weights of the model will be saved in ```data/models/{method}```.

## 4. Evaluation (Chapter 7)

In this part of the project (```src/sft/evaluation```) some metrics are computed for the trained LLM using a LLM judge. It could also be compared with the original LLM (i.e., the LLM without fine-tuning) in order to see if there are actually some improvements.

To run this step, just type in terminal

    python -m src.evaluation.evaluation

and a csv with the metrics will be saved in ```data/evaluation/evaluation.csv```.


## 5. RAG Inference Pipeline (Chapter 9)

In this part of the project (```src/rag_main```) the RAG is implemented for the inference with the LLM. It consists of three steps (pre-retrieval, retrieval and post-retrieval).

## Others

Note that chapters 1 and 2 are not included because they are just to understand the project and set up the environment; chapter 8 is not included because it just gives techniques to optimize the LLM inference; and chapters 10 and 11 are not included because they correspond to the model deployment.
