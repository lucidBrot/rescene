#!/usr/bin/env python3

# Goal: 
#       Given an input image, randomly generate a new one.
#       It should give some sense of being related to the original one.
#
# Future Goals:
#       Given a video, create a new one. Considering temporal consistency
#       and flow, to avoid flickering.
#
# Approach:
#       Just some ideas. Implement them. See how it goes. Worry about 
#       performance later. Just explore.
# 
# This attempt:
#       For every pixel, add its surrounding 5x5 kernel neighbours to a storage.
#       Choose a random color from the dict (FUTURE: other than uniform.) as
#       first pixel. Then maintain structure by going in direction only with
#       colors that are in that direction for some sufficiently similar original
#       color pixel. 
#       We could grow constantly in the same direction, but that would only 
#       maintain either horizontal or vertical structure.
#       We could grow in stepsize 2 vertically and then fill the remaining lines
#       only in the middle and grow them horizontally.
#       We could grow in random available directions by sampling from a set of 
#       already-chosen pixels and then from their neighbours. if is zero, remove
#       that pixel from possible choices.
#       I think the random option is the most promising one.
