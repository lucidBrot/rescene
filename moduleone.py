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

from dataclasses import dataclass
from PIL import Image
import numpy as np
import itertools

def load_numpy_image (infilename ):
    """
        https://stackoverflow.com/a/7769424/2550406
    """
    img = Image.open ( infilename )
    img.load()
    data = np.asarray ( img, dtype="int32" )
    return data

class Rescener():
    CANARY_UNSET = -1

    def __init__ (self, image_path, output_path, color_similarity_threshold = 4):
        self.image_path = image_path
        self.output_path = output_path
        # [H, W, RGB]
        self.image_np = load_numpy_image( image_path )
        self.color_similarity_threshold = color_similarity_threshold
        self.rng = np.random.default_rng()
        # [H, W, RGB]
        self.target_image = np.full_like(self.image_np, Rescener.CANARY_UNSET)

    def random_pixel_coords(self):
        """
            returns (h,w) in self.image_np coordinates
        """
        return self.rng.choice(self.image_np.shape[0], 1, replace=False),\
                self.rng.choice(self.image_np.shape[1], 1, replace=False)

    def neighbouring_coords_of(self, h, w):
        """
            returns a list of coordinates in range of the image that neighbour (h,w)
        """
        # concept for the general case with _range != 1
        # is to take the top and bottom sandwiches first and then fill up the middle.
        # unfinished because I am tired.
        ##l_h_top = { max(0, h - _range + row) for row in range(_range) }
        ##l_h_bottom = { min(self.image_np.shape[0]-1, h + _range - row) for row in range(_range) }
        ##l_w_full = set(range(max(0, w - _range),min(self.image_np.shape[1], w + _range + 1)))
        ##l_coords_top_bottom = itertools.product(l_h_top|l_h_bottom, l_w_full)

        # case where _range == 1
        potential_choices = [(h-1, w), (h, w-1), (h+1, w), (h, w+1)]
        H,W = self.image_np.shape[0], self.image_np.shape[1]
        actual_coords = [(y,x) for (y,x) in potential_choices if (-1 < x < W) and (-1 < y < H)]
        return actual_coords

    def self.get_colors_for_direction(direction, lookup_color):
        y,x = direction
        threshold = self.color_similarity_threshold
        # get the indices of the colors similar enough to the lookup color
        flat_candidate_coords = (np.linalg.norm((self.image_np - lookup_color[None, None, :], axis=2).view(-1) - threshold).nonzero()
        # return all colors to the $direction of the candidates. as a list.
        # take also uses the flat view when no axis is specified
        # The mode 'clip' means we repeat on the border.
        color_arr = np.take(self.image_np, flat_candidate_coords, mode='clip')
        return color_arr
        

    def start (self):
        # choose midpoint color randomly
        h, w = self.image_np.shape[0]//2, self.image_np.shape[1]//2
        self.target_image[h,w,:] = self.image_np[*self.random_pixel_coords(),:]
        # add it to our working set of borderset pixels
        self.borderset_pixels = [(h,w)]
        
        while self.borderset_pixels:
            # while not empty working set, do work on one pixel after the other. Randomly.
            idx = self.choice(len(self.borderset_pixels))
            h,w = self.borderset_pixels[idx]
            # If all neighbours are set, remove from working list and continue.
            # Else, choose a random one.
            neighbour_coords = [(y,x) for (y,x) in self.neighbouring_coords_of(h,w) if self.target_image[y,x] != Rescener.CANARY_UNSET]
            if len(neighbour_coords) == 0:
                self.borderset_pixels.remove((h,w))
                continue
            y,x = self.choice(neighbour_coords)
            # want to propagate a possible color by looking at every pixel with roughly
            # $CURRENT_COLOR and seeing what is there.
            direction = (y-h,x-w)
            color_options = self.get_colors_for_direction(direction)
            color_chosen = self.rng.choice(color_options)
            self.target_image[y,x,:] = color_chosen

            # add new option to working set.
            self.borderset_pixels.add((y,x))

        # TODO: save resulting image or at least show it.
        # TODO: debug
        # TODO: call this.
