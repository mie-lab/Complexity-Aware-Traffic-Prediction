rm -rf CATP/intermediate_folder
mkdir CATP/intermediate_folder
tar -czvf CATP.tar.gz CATP
echo "zipped CATP folder; moving to server"
scp CATP.tar.gz niskumar@kelut.sec.sg:
ssh -t  niskumar@kelut.sec.sg << EOF
  cd ~/NeurIPS2022-traffic4cast/exploration
  rm -rf CATP
  mv ~/CATP.tar.gz ./
  tar -xf CATP.tar.gz
  echo "Decompression complete and tar.gz deleted"
  echo "Logging out of server"
  exit
EOF
echo "Logged out of server"



