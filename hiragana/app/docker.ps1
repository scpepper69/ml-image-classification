function pwd_as_linux {
  "/$((pwd).Drive.Name.ToLowerInvariant())/$((pwd).Path.Replace('\', '/').Substring(3))"
}
docker run -d -it --name hiragana --rm -v "$(pwd_as_linux)/../learning:/models/" -p 9002:9002 sleepsonthefloor/graphpipe-tf:cpu --model=/models/hiragana.pb --listen=0.0.0.0:9002
