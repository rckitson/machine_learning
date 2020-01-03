# Traffic simulation

I came accross the topic of traffic simulation and wanted to create a quick
tool. The simulation is very rudimentary, a set of cars travelling in a circle.
The lead car in the video marked by an 'X' is originally too far behind. This causes
a cascade of 'braking' behind the car and meanwhile the lead car is trying to catch up the 
car ahead. 

As the video progresses you can see this 'wave' of information travelling through the set of
cars similar to what happens in the real world of stop and go traffic.

This was modeled as a series of masses in a chain connected by springs and dampers. I changed 
from a typical linear spring to a cubic spring that heavily penalizes the cars from getting to far
or too close to each other. Some tweaking was required on the 'stiffness' and 'damping' coefficients
to create a useful result.

[See the video](traffic.mp4)
