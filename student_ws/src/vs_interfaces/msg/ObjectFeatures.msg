# Image features of the target object, in pixels of the colour image.
std_msgs/Header header

bool valid
# "detector" when the value comes straight from a foundation model call, "tracker" otherwise
string source
string prompt

float64 u
float64 v
float64 area
# Orientation of the object's principal axis in the image [rad], 0 if only a box is known
float64 angle
# False if the angle carries no information (box only, or a round or square mask)
bool angle_valid
# Axis-aligned box: x_min, y_min, x_max, y_max
float64[4] bbox
float32 score

# Age of the last detector result and its inference time [s]
float64 detection_age
float64 detector_latency
