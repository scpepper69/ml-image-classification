docker run -d -it --name hiragana --rm -v "$PWD:/models/" -p 9003:9003 sleepsonthefloor/graphpipe-tf:cpu --model=/models/hiragana.pb --listen=0.0.0.0:9003 

