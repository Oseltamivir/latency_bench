!/bin/bash

mkdir actions-runner && cd actions-runner
curl -o actions-runner-linux-x64-2.329.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.329.0/actions-runner-linux-x64-2.329.0.tar.gz
tar xzf ./actions-runner-linux-x64-2.329.0.tar.gz

./config.sh \
  --url https://github.com/Oseltamivir/latency_bench \
  --token $TOKEN \
  --name latency-bench \
  --work _work \
  --labels "lambda,gpu,nvidia,cuda,latency-bench" \
  --unattended --replace
./run.sh

sudo ./svc.sh install
sudo ./svc.sh start