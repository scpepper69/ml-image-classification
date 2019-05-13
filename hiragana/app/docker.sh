docker run -d -it --name hiragana --rm -v "$PWD/../learning:/models/" -p 9002:9002 sleepsonthefloor/graphpipe-tf:cpu --model=/models/hiragana.pb --listen=0.0.0.0:9002

