I'm planning to do a low resolution boundary version of this project.
there are two aspect of variations that I want:
1. I want to test 0.5 deg and 1.5 deg version. 
    - 1.5 deg boundary dataset is already in /tmp3/b12902101/era5_tw_forecast_1.5deg
    - 0.5 deg boundary starts from 0.25 deg boundary, and spatially downsample it by factor 2 using average pooling. this will have only 1/4 data in comparison with 0.25 deg boundary.
2. I want two ways of applying the low resolution boundary on Aurora model.
    - first is to directly apply. For example, each pixel in 0.5 deg boundary will occupy 4 pixel area in aurora model's computational grid.
    - second is to use interpolation to turn the low resolution boundary into the same resolution as aurora model's computational grid. and then apply it on aurora model. 

    