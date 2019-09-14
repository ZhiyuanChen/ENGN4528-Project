# ENGN4528-2019-Major-Project
[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2Fyc-zc%2FENGN4528-Project.svg?type=shield)](https://app.fossa.io/projects/git%2Bgithub.com%2Fyc-zc%2FENGN4528-Project?ref=badge_shield)


##Useage

This project requires to use many python packages including but not limited to pika, Keras, TensorFlow-GPU, pytorch, 
pycocotools, etc. Please accept my sincere apologies for not providing a requirments.txt. Besides, you may also need to 
download the weights and put it into correct position. 

Please ensure you have updated the correct project path in the global package, and ensure the message queue address 
as well as username and password in the config.ini are correct. Please also make sure the config.ini is propaply 
configured, e.g., you may want to set the log level higher.

As this project is supposed to run distributedly, there are four master files and one capture file where they all can 
work independently. 

6 message queues would be declare once any of those file runs. They are compRequest, laneRequest, obstRequest, 
compResponse, laneResponse, obstResponse. 

To use this project, run all four of the master files, following statements shall be displayed/logged:

*listening queue on ip:port*

*ready to consume* 
 
 Please ensure the master file in the root directory is running on the machine you would like to display the results on.
 
 After all master files are running, run the capture file on the machine with pictures. It would capture screenshots of 
 720p range of the top left screen, crop and encode the screenshots and send it to all request queues accordingly. 
 
 At this point, the computer running the master file in root directory shall start displaying results on it's screen.

### Dockerfile useage

##### The dockfile has not been tested. Please do contact me if it does not work.

To use Dockerfile, you need to copy corresponding dockerfile to root directory. Please ensure the nvidia-docker is 
propaply installed. You may find this [web page](http://zc.int.xyz/kubernetes/install/) helpful.

## Master file definition

The master file in the root directory listens to all response queues and would created three windows on the screen to 
display the result from each response queue. 

The master file in comprehensive folder listen to compRequest, it shall receive the dashboard of the vehicle and publish
the result back to compResponse. 

The master file in lane folder listen to laneRequest, it shall receive the windshield of the image and publish
the result back to laneResponse. 

The master file in obstacle folder listen to obstRequest, it shall receive the windshield of the image and publish
the result back to obstResponse. 





## License
[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2Fyc-zc%2FENGN4528-Project.svg?type=large)](https://app.fossa.io/projects/git%2Bgithub.com%2Fyc-zc%2FENGN4528-Project?ref=badge_large)