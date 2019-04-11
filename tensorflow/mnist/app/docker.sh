docker run -d -it --name mnist --rm -v "$PWD:/models/" -p 9002:9002 sleepsonthefloor/graphpipe-tf:cpu --model=/models/expert-graph.pb --listen=0.0.0.0:9002 

