docker run -d -it --name japanese --rm -v "/home/apps/git/ml/tensorflow/japanese/learning:/models/" -p 9012:9012 sleepsonthefloor/graphpipe-tf:cpu --model=/models/japanese.pb --listen=0.0.0.0:9012
