rm -rf build 
rm -rf dist
rm -rf ENACT.egg-info
python setup.py build
python setup.py install