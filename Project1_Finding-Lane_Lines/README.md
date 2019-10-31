# **Finding Lane Lines on the Road** 

---

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

First, I calculate the lane line detection drawing (According to y = ax + b formula), and apply it to the pipeline.
My Pipeline includes five steps:
1. gray scale
2. Gaussian Smoothing
3. Canny
4. masked
5. Hough transform


### 2. Identify potential shortcomings with your current pipeline

One shortcoming is the lane line drawing might "blink" sometimes, 


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to build a stable line that won't "blink".



---
and the project will be done.

