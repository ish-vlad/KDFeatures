# KDFeatures

Structure
```python
 /datasets/
 |-> MNIST 2D/
     |-> Rotation/
            |-> 30/
                |-> pair-0001_rot-30.pkl
                |-> ...
            |-> ...
     |-> Translation/
            |-> 0.01/
                |-> pair-0001_trans-0.01.pkl
                |-> ...
            |-> ...
     |-> Rotation_and_Translation/
            |-> 30-0.01/
                |-> pair-0001_rot-30_trans-0.01.pkl
                |-> ...
            |-> ...
     |-> Noise/
            |-> 0.001/
                |-> pair-0001_rot-30_trans-0.01_noise-0.001.pkl
                |-> ...
            |-> 0.010/
                |-> pair-0001_rot-30_trans-0.01_noise-0.010.pkl
                |-> ...
            |-> ...
```

Each _pair-*.pkl_ consist of 9 elements:
1. [Key: **source_pc**] First pointcloud as matrix (256, 2)
2. [Key: **target_pc**] Second pointcloud as matrix (256, 2)
3. [Key: **true_T**] Ground truth transformation matrix (4, 4)
4. [Key: **source_KD_(0-6)**] TODO All KD-features of the first  PC
5. [Key: **source_KD_root**] TODO Root KD-features of the first PC
6. [Key: **target_KD_(0-6)**] TODO All KD-features of the second PC
7. [Key: **target_KD_root**] TODO Root KD-features of the second PC
8. [Key: **source_FPFH**] TODO FPFH features of the first PC
9. [Key: **target_FPFH**] TODO FPFH features of the second PC



```python
/results/
|-> Mnist 2D/
    |-> tables/
        |-> date.rot-30.csv
        |-> date.rot-60.csv
        |-> ...
    |-> pictures/
        |-> date.rot-30.pair-0001.png
        |-> ...
```

