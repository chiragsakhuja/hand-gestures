Project Palm 
========

For our final project, we tackled the problem of Hand detection. In this README we have outlined the way the directory is organized and the programs to run to make the code work

### Neccessary libraries, languages, and hardware 

* Python 2.7 
* Numpy 
* Open cv 2.4.9
* PyUserInput 
* Webcam attached to the computer 


### Structure of files and folders 

* Images: Has the images for progress reports and proposals 
* older_code: Has methods such as blob detection, haar_classifiers, which were not fully implmented or used 
* progress_reports: Has all the progress reports that were turned, including the powerpoint and proposal 
* test: Text files used for the unit testing 
* test_image and test_image_2: Test images used for running the unit tests
* unit_test: Units that we ran to find the accuracy and compile all our results 
* project_palm: Main file that contained the our code. 

#### Running the code 

```bash
python project_palm.py
```

When the window opens, it will show the hue detection region as a grid
of blue rectangles. To initiate the hue detection, place your palm over
the region, covering it entirely, and then press the 'q' key. The
program will analyze the hue in the region for 25 frames and then
automatically go into gesture mode. From this window, you can make
gestures to control the mouse.
