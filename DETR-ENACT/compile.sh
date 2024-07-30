rm -r models/ENACT_attn/build
rm -r models/ENACT_attn/dist
rm -r models/ENACT_attn/ENACT.egg-info
cd models/ENACT_attn
python setup.py build
python setup.py install 
cd ../..