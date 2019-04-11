docker run -d -it --rm -v "$PWD/../learning:/models/" -p 9004:9004 sleepsonthefloor/graphpipe-tf:cpu --model=/models/gface.pb --listen=0.0.0.0:9004 

