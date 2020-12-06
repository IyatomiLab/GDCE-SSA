#!/bin/sh

cd data

cd livedoor
curl -OL https://www.rondhuit.com/download/ldcc-20140209.tar.gz
tar -zxf ldcc-20140209.tar.gz

python make_data.py

cd ..

cd font

curl -OL http://moji.or.jp/wp-content/ipafont/IPAfont/ipag00303.zip
tar -zxf ipag00303.zip
mv ipag00303/ipag.ttf ./

cd ..

cd characters
python make_data.py

cd ..

python split_train_test.py