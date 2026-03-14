#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 14:34:01 2024

@author: erik
"""
import gdsfactory as gf
c = gf.Component()
c.add_polygon([(-8, -6), (6, 8), (7, 17), (9, 5)], layer=(1, 0))
c.plot()
c.show()