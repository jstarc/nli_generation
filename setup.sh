wget -P data "http://nlp.stanford.edu/projects/snli/snli_1.0.zip"
unzip data/snli_1.0.zip -d data
wget -P data "http://nlp.stanford.edu/data/glove.6B.zip"
unzip -j data/glove.6B.zip glove.6B.50d.txt -d data
python load_data.py repackage
