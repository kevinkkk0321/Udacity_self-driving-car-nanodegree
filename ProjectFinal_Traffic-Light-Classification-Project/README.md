# Traffic Light Classification

---

**Traffic Light Classification**

The goals/steps of this project are the following:

* Gather and label the datasets
* Transfer learning on a TensorFlow model
* Classify the state of traffic lights
* Summarize the results with a written report


Install tensorflow object detection api on AWS instance

1. clone tensorflow project
$ git clone https://github.com/tensorflow/models.git
$ cd models 
$ git checkout f7e99c0

2. install tensorflow-gpu and other settings
    sudo apt-get update
pip install tensorflow-gpu==1.4
sudo apt-get install protobuf-compiler python-pil python-lxml python-tk
cd models/research
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

3. sometimes occur errors
E: Could not get lock /var/lib/dpkg/lock - open (11: Resource temporarily unavailable)
E: Unable to lock the administration directory (/var/lib/dpkg/), is another process using it?
I'll reboot the instance and problem solved

4. run model_builder_test test if tensorflow correctly installed
python object_detection/builders/model_builder_test.py
if no error means tensorflow correctly installed

5. put this project on AWS，run following command and start training
$ python train.py --logtostderr --train_dir=./models/train --pipeline_config_path=./config/ssd_inception_v2_coco_sim.config
