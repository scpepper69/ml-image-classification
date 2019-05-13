docker run -d -it --name mnist --rm -v "$PWD/../learning:/models/" -p 9001:9001 sleepsonthefloor/graphpipe-tf:cpu --model=/models/expert-graph.pb --listen=0.0.0.0:9001
