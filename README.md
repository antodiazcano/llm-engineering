Complete LLM project: recollect data, store it into a db, use RAG and fine-tune the model. It is based on the "LLM Engineer's Book", wrote by Paul Iusztin and Maxime Labonne.

Poner que hay que instalar mongo db y comandos del vídeo de neural nine

iniciar mongodb

    mongosh
    sudo systemctl unmask mongod
    sudo service mongod start


para iniciar zenml es

    zenml login --local

Si da error al correr ZenML:
    
    pip install ["zenml0.75"]

o algo así (avisa por la terminal del comando exacto).

