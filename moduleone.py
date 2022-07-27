#!/usr/bin/env python3

# Goal: 
#       Given an input image, randomly generate a new one.
#       It should give some sense of being related to the original one.
#
# Future Goals:
#       Given a video, create a new one. Considering temporal consistency
#       and flow, to avoid flickering. This could be done by adding a probability for
#       consistency and if that one is taken then the pixel is retained/moved from last frame.
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
#
#       In the future, I could add retaining of structures. E.g. using edge detection. Or wider
#       pattern-matching.
# TODO: speedup
# TODO: option to pause and continue?
# TODO: cli arguments for input and output files.
# TODO: allow user to choose starting color, or even colors to give more bias to.

from dataclasses import dataclass
from PIL import Image
import numpy as np
import itertools
import cv2
import logging

def load_numpy_image (infilename ):
    """
        https://stackoverflow.com/a/7769424/2550406
    """
    img = Image.open ( infilename )
    img.load()
    data = np.asarray ( img, dtype="int32" )
    return data

class Rescener():
    CANARY_UNSET = 137

    def __init__ (self, image_path, output_path, color_similarity_threshold = 4):
        logging.basicConfig()
        self.logger = logging.getLogger("Rescener")
        self.logger.setLevel(logging.DEBUG)
        self.image_path = image_path
        self.output_path = output_path
        # [H, W, RGB]
        #self.image_np = load_numpy_image( image_path )
        self.image_np = cv2.imread(image_path)
        self.logger.debug(f"[init]: Original Image values: {self.image_np.max()=}")
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

    def get_colors_for_direction(self, direction, lookup_color):
        """
            returns a 3-tuple of lists. They are the color values.
        """
        y,x = direction
        threshold = self.color_similarity_threshold
        # get the indices of the colors similar enough to the lookup color
        pixelwise_colordistance = self.image_np - lookup_color[None, None, :]
        pixelwise_colordistance_norm = np.linalg.norm(pixelwise_colordistance, axis=2)
        flat_candidate_coords = np.flatnonzero(pixelwise_colordistance_norm < threshold)
        # use the direction from the candidate coord. Compute the new index in the flat array
        flat_candidate_coords = [coord + self.image_np.shape[1] * direction[0] + direction[1] for coord in flat_candidate_coords]

        # get the color of the candidate pixels as a three-array each by converting them to 2d access again.
        colors_0 = np.take(self.image_np[...,0], flat_candidate_coords, mode='clip')
        colors_1 = np.take(self.image_np[...,1], flat_candidate_coords, mode='clip')
        colors_2 = np.take(self.image_np[...,2], flat_candidate_coords, mode='clip')
        # take also uses the flat view when no axis is specified
        # The mode 'clip' means we repeat on the border.
        return (colors_0, colors_1, colors_2)

    def start (self, use_gui=True):
        # choose midpoint color randomly
        h, w = self.image_np.shape[0]//2, self.image_np.shape[1]//2
        rh, rw = self.random_pixel_coords()
        self.target_image[h,w,:] = self.image_np[rh, rw,:]
        # add it to our working set of borderset pixels
        self.borderset_pixels = {(h,w)}
        self.logger.debug(f"[start]: chose midpoint at {(h,w)} in image of size {self.image_np.shape}")
        
        num_pixels_colored_in_target = 1
        while self.borderset_pixels:
            # while not empty working set, do work on one pixel after the other. Randomly.
            h,w = self.rng.choice(tuple(self.borderset_pixels))
            # If all neighbours are set, remove from working list and continue.
            # Else, choose a random one.
            neighbour_coords = [(y,x) for (y,x) in self.neighbouring_coords_of(h,w) if (self.target_image[y,x] == Rescener.CANARY_UNSET).all()]
            if len(neighbour_coords) == 0:
                self.borderset_pixels.remove((h,w))
                self.logger.debug(f"[start]: removed pixel {(h,w)} from working set.")
                continue
            y,x = self.rng.choice(neighbour_coords)
            # want to propagate a possible color by looking at every pixel with roughly
            # $CURRENT_COLOR and seeing what is there.
            direction = (y-h,x-w)
            color_options = self.get_colors_for_direction(direction, lookup_color = self.target_image[h,w,:]) # a 3-tuple of lists.
            color_index = self.rng.choice(len(color_options[0]))
            color_chosen = np.c_[color_options[0][color_index], color_options[1][color_index], color_options[2][color_index]]
            self.target_image[y,x,:] = color_chosen
            self.logger.debug(f"[start]: {color_chosen=}")

            # add new option to working set.
            self.borderset_pixels.add((y,x))
            self.logger.debug(f"[start]: added pixel {(h,w)} to working set.")

            # update the shown image
            if use_gui:
                # TODO: replace the unset color values with something in color range?
                # because something is still broken here.
                cv2.imshow('original', self.image_np)
                cv2.imshow('target', self.target_image)
                cv2.waitKey(0)
                self.logger.debug(f"[start]: Displaying Image.")

            if num_pixels_colored_in_target % 10 == 0:
                self.logger.info(f"[start]: Saving wip at {num_pixels_colored_in_target}/{np.product(self.target_image.shape[:2])}")
                cv2.imwrite(self.output_path, self.target_image)
                cv2.imshow("target", self.target_image)
                cv2.waitKey(1) # seems to be needed.
                self.logger.debug(f"[init]: wip Image values: {self.target_image.min()=},{self.target_image.max()=}")

            self.logger.debug(f"[start]: Progress: {num_pixels_colored_in_target}/{np.product(self.target_image.shape[:2])}")
            num_pixels_colored_in_target += 1

        # save resulting image.
        cv2.imwrite(self.output_path, self.target_image)

def main():
    rescener = Rescener( image_path = "./input_image_fire.png",
                output_path = "./output_image_fire2.png",
                color_similarity_threshold = 4)
    rescener.start(use_gui=False)

if __name__ == "__main__":
    main()
