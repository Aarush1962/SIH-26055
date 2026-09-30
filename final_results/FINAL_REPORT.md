# Final End-to-End EW Scheduler Report

## Pipeline

Raw HDF5 → in-memory RF environment → frozen recursive belief updater → frozen Experiment-52 Neural scheduler. Oracle is computed independently in memory for post-run evaluation only.

## Aggregate results

| Policy | Micro ISR | Mean delay (s) |
|---|---:|---:|
| Random | 13.84% | 0.6341 |
| Greedy belief | 1.10% | 0.0000 |
| Experiment-52 Neural | 24.56% | 0.2864 |
| Oracle | 88.43% | 0.0259 |

- Experiment-52 Neural vs Random relative ISR gain: 77.44%
- Neural / Oracle event capture: 27.77%
- Total transmission events: 2818
- Test files: 9

## Per-file results

| file          |   total_events |   random_isr |   greedy_isr |   neural_isr |   oracle_isr |   neural_vs_oracle_capture |   neural_mean_delay_s |   oracle_mean_delay_s |
|:--------------|---------------:|-------------:|-------------:|-------------:|-------------:|---------------------------:|----------------------:|----------------------:|
| config_0.h5   |             24 |       0.3333 |       0.1667 |       0.4167 |       1.0000 |                     0.4167 |                0.7000 |                0.0063 |
| config_1.h5   |            234 |       0.2222 |       0.0513 |       0.4957 |       0.9744 |                     0.5088 |                0.1616 |                0.0154 |
| config_10.h5  |            280 |       0.1679 |       0.0179 |       0.2643 |       0.9607 |                     0.2751 |                0.2703 |                0.0136 |
| config_100.h5 |            420 |       0.1643 |       0.0024 |       0.2119 |       0.9429 |                     0.2247 |                0.1961 |                0.0520 |
| config_101.h5 |            172 |       0.1977 |       0.0058 |       0.4186 |       0.9477 |                     0.4417 |                0.2118 |                0.0123 |
| config_102.h5 |            365 |       0.1260 |       0.0055 |       0.1644 |       0.8521 |                     0.1929 |                0.3258 |                0.0186 |
| config_103.h5 |            177 |       0.1864 |       0.0113 |       0.2768 |       0.9266 |                     0.2988 |                0.7878 |                0.0091 |
| config_104.h5 |            472 |       0.0890 |       0.0021 |       0.2373 |       0.8708 |                     0.2725 |                0.3839 |                0.0197 |
| config_105.h5 |            674 |       0.0875 |       0.0045 |       0.1632 |       0.7804 |                     0.2091 |                0.1691 |                0.0365 |

## Event-duration breakdown

| duration_group   |    events |   neural_intercepted |   oracle_intercepted |   neural_isr |   oracle_isr |
|:-----------------|----------:|---------------------:|---------------------:|-------------:|-------------:|
| 5+ steps         |  756.0000 |             407.0000 |             750.0000 |       0.5384 |       0.9921 |
| 3-4 steps        |  405.0000 |             105.0000 |             386.0000 |       0.2593 |       0.9531 |
| 2 steps          |  513.0000 |              77.0000 |             463.0000 |       0.1501 |       0.9025 |
| 1 step           | 1144.0000 |             103.0000 |             893.0000 |       0.0900 |       0.7806 |