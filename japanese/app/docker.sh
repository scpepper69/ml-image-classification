docker run -d -it --name japanese --rm -v "$PWD/../learning:/models/" -p 9003:9003 sleepsonthefloor/graphpipe-tf:cpu --model=/models/japanese.pb --listen=0.0.0.0:9003
