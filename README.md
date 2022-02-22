# imagenet-index-test

This program is too show the index issues which appear to be in the pentathlon dataset. Getting the features for both SENet and ResNeXt from a single clip of a cat do return very different results.

Run this by:
```
git clone git@github.com:RSteendam/imagenet-index-test.git`
cd imagenet-index-test
python3 -m pip install -r requirements.txt
python3 main.py
```
Expected output
```
TESTING IMAGENET CLASSES ON RESNEXT AND SENET

clip: a4209223a8

SENTENCES:
a hand descends into frame from the left side of the screen.
someone reaches their hand in and pets the cat.
the camerawoman begins to pet the cat
the cat is pet
first time we see a hand pet the cat

PREDICTIONS:
model: resnext101_32x48d-avg-logits.pickle
cock 0.8212337493896484
white stork 0.10772242397069931
orangutan 0.024799110367894173
espresso maker 0.02384907193481922
eft 0.006344108376652002

model: resnext101_32x48d-max-logits.pickle
cock 0.6316341161727905
orangutan 0.1523888260126114
squirrel monkey 0.0708673819899559
espresso maker 0.04539934918284416
white stork 0.038230083882808685

model: senet154-avg-logits.pickle
quilt 0.7298898696899414
unicycle 0.11593735218048096
iron 0.030961280688643456
thimble 0.00709990318864584
Arabian camel 0.0018684705719351768

model: senet154-max-logits.pickle
quilt 0.6595724821090698
iron 0.11551545560359955
unicycle 0.0865170806646347
thimble 0.041349511593580246
Arabian camel 0.016129571944475174

END
```
