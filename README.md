# ENGN4528-2019-major-project


## Data transmitting using Message Queue
class Master in globals packaged message queue and log. 
Inherit Master class, init the basic_consume and start_consuming and rewrite receive for further process.


##### Status code and message

|Code Range |Allocation |
|-----------|-----------|
|100        |received   |
|200        |success    |
|201-399    |Central    |
|400-599    |Sign       |
|600-799    |Obstacle   |
|800-999    |Lane       |

Status code for central

|Code       |Message    |
|-----------|-----------|
|210-219    |Failed to capture screenshot|
|220-229    |Failed to send screenshot|
|300-309    |Failed to load json|
|300        |Illegal message|
|301        |Failed to receive message|
|306        |Failed to load message|
|307        |Failed to load image|


