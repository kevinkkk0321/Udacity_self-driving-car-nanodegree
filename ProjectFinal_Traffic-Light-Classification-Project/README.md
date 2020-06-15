# Traffic Light Classification

---

**Traffic Light Classification**

The goals/steps of this project are the following:

* Gather and label the datasets
* Transfer learning on a TensorFlow model
* Classify the state of traffic lights
* Summarize the results with a written report


Install tensorflow object detection api on AWS instance

1. 主要是clone出project
$ git clone https://github.com/tensorflow/models.git
$ cd models 
$ git checkout f7e99c0

2. 安裝tensorflow-gpu和其他環境設定
sudo apt-get update
pip install tensorflow-gpu==1.4
sudo apt-get install protobuf-compiler python-pil python-lxml python-tk
cd models/research
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

3. 在執行時有時會遇到
E: Could not get lock /var/lib/dpkg/lock - open (11: Resource temporarily unavailable)
E: Unable to lock the administration directory (/var/lib/dpkg/), is another process using it?
這時我都直接reboot來解決

4. 最後執行
python object_detection/builders/model_builder_test.py
若沒出現錯誤則表示成功安裝

5. 把此project放到AWS上，執行 開始訓練
$ python train.py --logtostderr --train_dir=./models/train --pipeline_config_path=./config/ssd_inception_v2_coco_sim.config
