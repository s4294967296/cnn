# STL imports
import os
import logging
import configparser

# 3rd party imports

# Home-made :) imports

########################################################################################################################
# CELFA - CNN Evaluation Library For ANNIE
# ----------------------------------------
# This module simply encapsulates all Error classes.
########################################################################################################################

########################################################################################################################
#
# Exceptions concerning data and data handling
#
########################################################################################################################


class ErrorMismatch(Exception):
    """Raised when a process depends on two separate data files with the same index, but not both could be loaded"""
    pass


class ErrorParameter(Exception):
    """No fitting parameters have been provided."""
    pass

########################################################################################################################
#
# Exceptions concerning the neural net and neural net creation
#
########################################################################################################################


########################################################################################################################
#
# Exceptions concerning neural net performance and evaluation
#
########################################################################################################################

