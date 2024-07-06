conda create --name medrag python=3.10 -y
conda activate medrag
pip install -r requirements.txt

# download pubmed data
wget https://knowledge-bottlenecks.s3.amazonaws.com/pubmed_all.zip
mkdir corpus
unzip pubmed_all.zip -d corpus/
rm pubmed_all.zip