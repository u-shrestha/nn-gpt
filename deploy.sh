git clone https://github.com/480284856/transformers.git -b fix-save-pretrained-quantized-models
git clone https://github.com/480284856/nn-gpt.git -b gu/gpt-oss

cd nn-gpt
pip install -e .
cd ../transformers
pip install -e .
pip install deepspeed==0.18.3


cd ../nn-gpt
python -m ab.gpt.TuneNNGen_20B_oss